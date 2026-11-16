from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config
from .io import sha256

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD = (0.213, 0.156, 0.143)
POSITIVE_PROMPTS = (
    "a satellite image of a ground-mounted solar power plant",
    "photovoltaic panel arrays in a satellite image",
    "a utility-scale solar farm seen from above",
    "rows of ground-mounted solar panels",
)
NEGATIVE_CLASSES = (
    "agricultural greenhouses", "industrial metal roofs", "water bodies and fish ponds",
    "cropland fields", "bare soil and desert", "urban buildings and roads",
)


def _normalize_rgb(rgb: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
    x = rgb.float() / 255.0
    return (x - x.new_tensor(mean)[None, :, None, None]) / x.new_tensor(std)[None, :, None, None]


def _frozen(model: nn.Module) -> None:
    if any(m.training for m in model.modules()) or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("Encoder must be fully eval/frozen with no gradients")


class GeoRSCLIP:
    def __init__(self, weights: Path, expected_sha: str, device: str):
        import open_clip

        digest = sha256(weights)
        if digest != expected_sha:
            raise ValueError(f"GeoRSCLIP checkpoint SHA-256 mismatch: {digest}")
        self.device = device
        self.open_clip_version = getattr(open_clip, "__version__", "unknown")
        self.model = open_clip.create_model("ViT-B/32", pretrained=None, force_quick_gelu=True)
        try:
            state = torch.load(weights, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weights, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError("Expected direct GeoRSCLIP state_dict")
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        _frozen(self.model)
        visual = self.model.visual
        if not all(hasattr(visual, key) for key in ("conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "ln_post", "proj")):
            raise RuntimeError("Unsupported OpenCLIP ViT visual interface")
        if visual.conv1.kernel_size != (32, 32) or visual.conv1.stride != (32, 32):
            raise RuntimeError("GeoRSCLIP is not a ViT-B/32 patch encoder")
        if visual.positional_embedding.shape != (50, 768):
            raise RuntimeError("Expected native 7x7 patch positional embedding")
        self.checkpoint_sha256 = digest
        self._verify_cls()
        self.text_positive, self.text_negative = self._texts(open_clip)

    @torch.no_grad()
    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        visual = self.model.visual
        patch = visual.conv1(x).flatten(2).transpose(1, 2)
        size = int(math.isqrt(patch.shape[1]))
        if size * size != patch.shape[1]:
            raise RuntimeError("Non-square visual token grid")
        cls = visual.class_embedding.to(patch.dtype).expand(x.shape[0], 1, -1)
        pos = visual.positional_embedding.to(patch.dtype)
        if size != 7:
            grid = pos[1:].reshape(1, 7, 7, -1).permute(0, 3, 1, 2)
            grid = F.interpolate(grid.float(), size=(size, size), mode="bicubic", align_corners=False)
            pos = torch.cat((pos[:1], grid.to(patch.dtype).permute(0, 2, 3, 1).reshape(-1, pos.shape[-1])), dim=0)
        tokens = visual.ln_pre(torch.cat((cls, patch), dim=1) + pos)
        tokens = visual.transformer(tokens)
        tokens = visual.ln_post(tokens)
        return tokens @ visual.proj

    @torch.no_grad()
    def _verify_cls(self) -> None:
        probe = torch.arange(3 * 224 * 224, device=self.device, dtype=torch.float32).reshape(1, 3, 224, 224)
        probe = ((probe.remainder(251)) / 250).float()
        direct = self.model.encode_image(probe).float()
        dense = self._tokens(probe)[:, 0].float()
        if not torch.allclose(direct, dense, rtol=1e-4, atol=1e-4):
            raise RuntimeError("Dense GeoRSCLIP path does not reproduce native CLS at 224")

    @torch.no_grad()
    def _texts(self, open_clip) -> tuple[torch.Tensor, torch.Tensor]:
        prompts = POSITIVE_PROMPTS + tuple("a satellite image of " + name for name in NEGATIVE_CLASSES)
        ids = open_clip.tokenize(list(prompts), context_length=77).to(self.device)
        vectors = F.normalize(self.model.encode_text(ids).float(), dim=-1)
        positive = F.normalize(vectors[:len(POSITIVE_PROMPTS)].mean(0), dim=0)
        return positive, vectors[len(POSITIVE_PROMPTS):]

    @torch.no_grad()
    def encode(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _frozen(self.model)
        x = _normalize_rgb(rgb.to(self.device), CLIP_MEAN, CLIP_STD)
        if x.shape[-2:] != (512, 512):
            raise ValueError("GeoRSCLIP input must be 512x512")
        features = F.normalize(self._tokens(x).float(), dim=-1)
        if features.shape != (x.shape[0], 257, 512) or not torch.isfinite(features).all():
            raise RuntimeError("Invalid GeoRSCLIP dense feature layout")
        positive = features @ self.text_positive
        negative = (features @ self.text_negative.T).amax(dim=-1)
        dense = (positive[:, 1:] - negative[:, 1:]).reshape(x.shape[0], 1, 16, 16)
        dense = F.interpolate(dense, size=(32, 32), mode="bilinear", align_corners=False)
        cls_pos = positive[:, :1] / 0.07
        cls_neg = (features[:, :1] @ self.text_negative.T).squeeze(1) / 0.07
        log_odds = cls_pos.squeeze(1) - torch.logsumexp(cls_neg, dim=1)
        return dense, log_odds


class DinoSAT:
    def __init__(self, source: Path, weights: Path, expected_prefix: str, device: str):
        digest = sha256(weights)
        if not digest.startswith(expected_prefix):
            raise ValueError(f"DINOv3 checkpoint SHA-256 mismatch: {digest}")
        if not (source / "dinov3" / "hub" / "backbones.py").is_file():
            raise FileNotFoundError("DINOv3 source directory lacks dinov3/hub/backbones.py")
        sys.path.insert(0, str(source.resolve()))
        backbones = importlib.import_module("dinov3.hub.backbones")
        if Path(backbones.__file__).resolve() != (source / "dinov3" / "hub" / "backbones.py").resolve():
            raise RuntimeError("Imported DINOv3 module is not from the configured source directory")
        self.model = backbones.dinov3_vitl16(pretrained=True, weights=str(weights.resolve()))
        self.model.to(device).eval().requires_grad_(False)
        _frozen(self.model)
        if not hasattr(self.model, "blocks") or len(self.model.blocks) != 24:
            raise RuntimeError("Expected 24 DINOv3 ViT-L transformer blocks")
        self.layer_indices = tuple(range(20, 24))
        self.device, self.checkpoint_sha256 = device, digest

    @torch.no_grad()
    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        _frozen(self.model)
        x = _normalize_rgb(rgb.to(self.device), SAT_MEAN, SAT_STD)
        layers = self.model.get_intermediate_layers(x, n=4, reshape=True, norm=True)
        if len(layers) != 4 or any(t.shape != (x.shape[0], 1024, 32, 32) for t in layers):
            raise RuntimeError("DINOv3 four-layer API or patch grid differs from registered contract")
        result = torch.stack([layer.float() for layer in layers]).mean(0)
        if not torch.isfinite(result).all():
            raise RuntimeError("Non-finite DINO feature")
        return result


def load_encoders(config: Config) -> tuple[GeoRSCLIP, DinoSAT]:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return (GeoRSCLIP(config.georsclip_weights, config.georsclip_sha256, config.device),
            DinoSAT(config.dinov3_source, config.dinov3_weights, config.dinov3_sha256_prefix, config.device))
