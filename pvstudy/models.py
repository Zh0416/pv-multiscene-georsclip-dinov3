from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def soft_dice(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p, y = probability.flatten(1), target.flatten(1)
    return (1 - (2 * (p * y).sum(1) + 1) / (p.sum(1) + y.sum(1) + 1)).mean()


def fusion_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    clipped = probability.clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(clipped, target) + soft_dice(clipped, target)


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + soft_dice(torch.sigmoid(logits), target)


class FusionGate(nn.Module):
    def __init__(self, mode: str = "C6", theta: float = 0.5):
        super().__init__()
        if mode not in ("C5", "C5g", "C6"):
            raise ValueError("Unsupported fusion mode")
        self.mode, self.theta = mode, theta
        self.context = nn.Sequential(nn.Conv2d(1024, 16, 1), nn.GroupNorm(4, 16), nn.GELU())
        self.gate = nn.Sequential(nn.Conv2d(22, 16, 3, padding=1), nn.GroupNorm(4, 16),
                                  nn.GELU(), nn.Conv2d(16, 2, 1))
        nn.init.constant_(self.gate[-1].bias[1], -1.0)

    def forward(self, dino: torch.Tensor, p_sem: torch.Tensor, p_spa: torch.Tensor,
                s_global: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if p_sem.shape != p_spa.shape or p_sem.shape[1:] != (1, 32, 32) or dino.shape[1:] != (1024, 32, 32):
            raise ValueError("Fusion feature shape mismatch")
        smooth = F.avg_pool2d(F.pad(p_sem, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        region = 0.15 + 0.85 * torch.sigmoid(10 * (smooth - self.theta))
        global_map = s_global.reshape(-1, 1, 1, 1).expand_as(p_sem)
        context = self.context(dino)
        inputs = torch.cat((p_sem, p_spa, region, region * p_spa,
                            (p_sem-p_spa).abs(), context, global_map), dim=1)
        gates = torch.sigmoid(self.gate(inputs))
        a, rho = gates[:, :1], gates[:, 1:]
        if self.mode == "C5":
            m = a * p_sem + (1 - a) * p_spa
            weights = (a, torch.zeros_like(a), 1 - a)
        elif self.mode == "C5g":
            m = a * p_sem + (1 - a) * region * p_spa
            weights = (a, 1 - a, torch.zeros_like(a))
        else:
            m = a * p_sem + (1 - a) * p_spa * (region + (1 - region) * rho)
            weights = (a, (1 - a) * (1 - rho), (1 - a) * rho)
        return m, {"a": a, "rho": rho, "R": region, "route": torch.cat(weights, dim=1)}


def local_indices(query_hw: int = 64, key_hw: int = 32, window: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    if query_hw * 8 != key_hw * 16 or window not in (3, 5):
        raise ValueError("Unsupported query/key geometry or window")
    qy, qx = torch.meshgrid(torch.arange(query_hw), torch.arange(query_hw), indexing="ij")
    center_y = ((8 * qy + 4) // 16).reshape(-1, 1)
    center_x = ((8 * qx + 4) // 16).reshape(-1, 1)
    radius = window // 2
    dy, dx = torch.meshgrid(torch.arange(-radius, radius + 1), torch.arange(-radius, radius + 1), indexing="ij")
    ky, kx = center_y + dy.reshape(1, -1), center_x + dx.reshape(1, -1)
    valid = (ky >= 0) & (ky < key_hw) & (kx >= 0) & (kx < key_hw)
    indices = ky.clamp(0, key_hw - 1) * key_hw + kx.clamp(0, key_hw - 1)
    return indices.long(), valid


class LocalCrossAttention(nn.Module):
    def __init__(self, window: int = 3):
        super().__init__()
        self.query = nn.Conv2d(64, 64, 1)
        self.key = nn.Conv2d(1024, 64, 1)
        self.value = nn.Conv2d(1024, 64, 1)
        self.output = nn.Conv2d(64, 64, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.lambda_raw = nn.Parameter(torch.tensor(math.log(math.expm1(0.1))))
        indices, valid = local_indices(window=window)
        self.register_buffer("indices", indices, persistent=False)
        self.register_buffer("valid", valid, persistent=False)

    def forward(self, query: torch.Tensor, dino: torch.Tensor, m: torch.Tensor,
                bias_enabled: bool) -> torch.Tensor:
        batch = query.shape[0]
        q = self.query(query).flatten(2).transpose(1, 2).reshape(batch, 4096, 4, 16)
        k = self.key(dino).flatten(2).transpose(1, 2)[:, self.indices].reshape(batch, 4096, -1, 4, 16)
        v = self.value(dino).flatten(2).transpose(1, 2)[:, self.indices].reshape(batch, 4096, -1, 4, 16)
        scores = (q[:, :, None] * k).sum(-1) / 4.0
        if bias_enabled:
            key_m = m.detach().flatten(2).transpose(1, 2)[:, self.indices, 0]
            query_m = F.interpolate(m.detach(), size=(64, 64), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
            scores = scores + F.softplus(self.lambda_raw) * query_m[:, :, None] * key_m[:, :, :, None]
        scores = scores.masked_fill(~self.valid[None, :, :, None], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=2)
        result = (attention[..., None] * v).sum(2).reshape(batch, 4096, 64)
        result = result.transpose(1, 2).reshape(batch, 64, 64, 64)
        return query + self.gamma * self.output(result)


def _block(input_channels: int, output_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1),
                         nn.GroupNorm(4, output_channels), nn.GELU(),
                         nn.Conv2d(output_channels, output_channels, 3, padding=1),
                         nn.GroupNorm(4, output_channels), nn.GELU())


class Decoder(nn.Module):
    def __init__(self, window: int = 3):
        super().__init__()
        self.e2 = _block(3, 16, 2)
        self.e4 = _block(16, 32, 2)
        self.e8 = _block(32, 64, 2)
        self.dino_query = nn.Conv2d(1024, 64, 1)
        self.rgb_query = nn.Conv2d(64, 64, 1)
        self.rgb_scale = nn.Parameter(torch.tensor(0.1))
        self.attention = LocalCrossAttention(window)
        self.d4 = _block(64 + 32 + 1, 64)
        self.d2 = _block(64 + 16 + 1, 32)
        self.d1 = _block(32 + 1, 16)
        self.logits = nn.Conv2d(16, 1, 1)

    def forward(self, rgb: torch.Tensor, dino: torch.Tensor, m: torch.Tensor,
                bias_enabled: bool = False) -> torch.Tensor:
        if rgb.shape[1:] != (3, 512, 512) or dino.shape[1:] != (1024, 32, 32) or m.shape[1:] != (1, 32, 32):
            raise ValueError("Decoder input must be RGB512, DINO32, M32")
        e2 = self.e2(rgb)
        e4 = self.e4(e2)
        e8 = self.e8(e4)
        q = F.interpolate(self.dino_query(dino), size=(64, 64), mode="bilinear", align_corners=False)
        q = q + self.rgb_scale * self.rgb_query(e8)
        z = self.attention(q, dino, m.detach(), bias_enabled)
        def up(x: torch.Tensor, hw: int) -> torch.Tensor:
            return F.interpolate(x, size=(hw, hw), mode="bilinear", align_corners=False)
        d4 = self.d4(torch.cat((up(z, 128), e4, up(m.detach(), 128)), dim=1))
        d2 = self.d2(torch.cat((up(d4, 256), e2, up(m.detach(), 256)), dim=1))
        return self.logits(self.d1(torch.cat((up(d2, 512), up(m.detach(), 512)), dim=1)))
