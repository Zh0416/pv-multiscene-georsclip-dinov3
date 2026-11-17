from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class GeoGeometry:
    model: str
    native: int
    patch: int
    width: int
    embed: int
    layers: int
    window_starts: tuple[int, ...]
    window_batch: int

    @property
    def native_grid(self) -> int:
        return self.native // self.patch

    @property
    def dense_grid(self) -> int:
        return math.ceil(512 / self.patch)

    @property
    def padded_size(self) -> int:
        return self.dense_grid * self.patch

    @property
    def dense_description(self) -> str:
        if self.patch == 32:
            return "native patch tokens, bicubic 7x7 positional interpolation to 16x16"
        return "RGB512 replicate-pad right/bottom to 518; bicubic 24x24 positions to 37x37; bilinear to 32x32"


GEOMETRIES = {
    "ViT-B/32": GeoGeometry("ViT-B/32", 224, 32, 768, 512, 12, (0, 96, 192, 288), 4),
    "ViT-L-14-336": GeoGeometry("ViT-L-14-336", 336, 14, 1024, 768, 24, (0, 98, 182), 2),
}


def _normalize_rgb(rgb: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
    x = rgb.float() / 255.0
    return (x - x.new_tensor(mean)[None, :, None, None]) / x.new_tensor(std)[None, :, None, None]


def _frozen(model: nn.Module) -> None:
    if any(m.training for m in model.modules()) or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("Encoder must be fully eval/frozen with no gradients")


class GeoRSCLIP:
    def __init__(self, weights: Path, expected_sha: str, device: str, model_name: str = "ViT-B/32"):
        import open_clip

        self.geometry = GEOMETRIES[model_name]
        digest = sha256(weights)
        if digest != expected_sha:
            raise ValueError(f"GeoRSCLIP checkpoint SHA-256 mismatch: {digest}")
        self.device = device
        self.open_clip_version = getattr(open_clip, "__version__", "unknown")
        self.model = open_clip.create_model(model_name, pretrained=None, force_quick_gelu=True)
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
        geometry = self.geometry
        if (visual.conv1.kernel_size != (geometry.patch, geometry.patch) or
            visual.conv1.stride != (geometry.patch, geometry.patch) or
            visual.positional_embedding.shape != (1 + geometry.native_grid ** 2, geometry.width) or
            visual.proj.shape != (geometry.width, geometry.embed) or
            len(visual.transformer.resblocks) != geometry.layers):
            raise RuntimeError(f"GeoRSCLIP visual architecture differs from {model_name}")
        self.checkpoint_sha256 = digest
        self._verify_cls()
        self.text_positive, self.text_negative = self._texts(open_clip)

    def _geometry(self) -> GeoGeometry:
        return getattr(self, "geometry", GEOMETRIES["ViT-B/32"])

    def _dense_input(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.shape[-2:] != (512, 512):
            raise ValueError("GeoRSCLIP input must be 512x512")
        x = _normalize_rgb(rgb.to(self.device), CLIP_MEAN, CLIP_STD)
        pad = self._geometry().padded_size - 512
        return F.pad(x, (0, pad, 0, pad), mode="replicate") if pad else x

    def _embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        visual, geometry = self.model.visual, self._geometry()
        patch = visual.conv1(x).flatten(2).transpose(1, 2)
        size = int(math.isqrt(patch.shape[1]))
        if size * size != patch.shape[1]:
            raise RuntimeError("Non-square visual token grid")
        cls = visual.class_embedding.to(patch.dtype).expand(x.shape[0], 1, -1)
        pos = visual.positional_embedding.to(patch.dtype)
        if size != geometry.native_grid:
            grid = pos[1:].reshape(1, geometry.native_grid, geometry.native_grid, -1).permute(0, 3, 1, 2)
            grid = F.interpolate(grid.float(), size=(size, size), mode="bicubic", align_corners=False)
            pos = torch.cat((pos[:1], grid.to(patch.dtype).permute(0, 2, 3, 1).reshape(-1, pos.shape[-1])), dim=0)
        return visual.ln_pre(torch.cat((cls, patch), dim=1) + pos)

    @torch.no_grad()
    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        visual = self.model.visual
        tokens = self._embed_tokens(x)
        if getattr(self, "_sequence_first", True):
            tokens = visual.transformer(tokens.transpose(0, 1)).transpose(0, 1)
        else:
            tokens = visual.transformer(tokens)
        tokens = visual.ln_post(tokens)
        return tokens @ visual.proj

    @torch.no_grad()
    def _verify_cls(self) -> None:
        native = self._geometry().native
        probe = torch.arange(3 * native * native, device=self.device, dtype=torch.float32).reshape(1, 3, native, native)
        probe = ((probe.remainder(251)) / 250).float()
        direct = self.model.encode_image(probe).float()
        dense = self._tokens(probe)[:, 0].float()
        if not torch.allclose(direct, dense, rtol=1e-4, atol=1e-4):
            self._sequence_first = False
            dense = self._tokens(probe)[:, 0].float()
            if not torch.allclose(direct, dense, rtol=1e-4, atol=1e-4):
                raise RuntimeError(f"Dense GeoRSCLIP path does not reproduce native CLS at {native}")

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
        geometry = self._geometry()
        x = self._dense_input(rgb)
        features = F.normalize(self._tokens(x).float(), dim=-1)
        if (features.shape != (x.shape[0], 1 + geometry.dense_grid ** 2, geometry.embed) or
            not torch.isfinite(features).all()):
            raise RuntimeError("Invalid GeoRSCLIP dense feature layout")
        positive = features @ self.text_positive
        negative = (features @ self.text_negative.T).amax(dim=-1)
        dense = (positive[:, 1:] - negative[:, 1:]).reshape(x.shape[0], 1, geometry.dense_grid, geometry.dense_grid)
        dense = F.interpolate(dense, size=(32, 32), mode="bilinear", align_corners=False)
        cls_pos = positive[:, :1] / 0.07
        cls_neg = (features[:, :1] @ self.text_negative.T).squeeze(1) / 0.07
        log_odds = cls_pos.squeeze(1) - torch.logsumexp(cls_neg, dim=1)
        return dense, log_odds

    @torch.no_grad()
    def encode_sliding224(self, rgb: torch.Tensor) -> torch.Tensor:
        """Legacy E05 name for the B/32 native-window score."""
        if self._geometry().model != "ViT-B/32":
            raise ValueError("Use encode_sliding_native for ViT-L-14-336")
        return self.encode_sliding_native(rgb)

    @torch.no_grad()
    def encode_sliding_native(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.encode_semantic_pair(rgb, sliding_native=True)[0]

    @torch.no_grad()
    def encode_semantic_pair(self, rgb: torch.Tensor, *, sliding_native: bool = False,
                             sliding224: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hard-negative margin and positive-only cosine on the same patch geometry."""
        _frozen(self.model)
        if sliding_native and sliding224:
            raise ValueError("Choose one sliding-window argument")
        if rgb.shape != (1, 3, 512, 512):
            raise ValueError("E03 expects one RGB512 image")
        geometry = self._geometry()
        x = self._dense_input(rgb)
        if sliding_native or sliding224:
            locations = [(y, x0) for y in geometry.window_starts for x0 in geometry.window_starts]
            total = torch.zeros((1, 2, geometry.dense_grid, geometry.dense_grid), device=x.device)
            counts = torch.zeros((1, 1, geometry.dense_grid, geometry.dense_grid), device=x.device)
            for start in range(0, len(locations), geometry.window_batch):
                chunk = locations[start:start + geometry.window_batch]
                crops = torch.cat([x[:, :, y:y + geometry.native, x0:x0 + geometry.native] for y, x0 in chunk])
                features = F.normalize(self._tokens(crops).float(), dim=-1)
                if (features.shape != (len(chunk), 1 + geometry.native_grid ** 2, geometry.embed) or
                    not torch.isfinite(features).all()):
                    raise RuntimeError("Invalid E03 sliding token layout")
                positive = features @ self.text_positive
                margin = positive - (features @ self.text_negative.T).amax(-1)
                maps = torch.stack((margin[:, 1:], positive[:, 1:]), dim=1).reshape(
                    len(chunk), 2, geometry.native_grid, geometry.native_grid)
                for i, (y, x0) in enumerate(chunk):
                    sy, sx = y // geometry.patch, x0 // geometry.patch
                    n = geometry.native_grid
                    total[0, :, sy:sy + n, sx:sx + n] += maps[i]
                    counts[0, :, sy:sy + n, sx:sx + n] += 1
            if not (counts > 0).all():
                raise RuntimeError("E03 sliding windows left uncovered patches")
            scores = total / counts
        else:
            features = F.normalize(self._tokens(x).float(), dim=-1)
            if (features.shape != (1, 1 + geometry.dense_grid ** 2, geometry.embed) or
                not torch.isfinite(features).all()):
                raise RuntimeError("Invalid E03 native token layout")
            positive = features @ self.text_positive
            margin = positive - (features @ self.text_negative.T).amax(-1)
            scores = torch.stack((margin[:, 1:], positive[:, 1:]), dim=1).reshape(
                1, 2, geometry.dense_grid, geometry.dense_grid)
        scores = F.interpolate(scores, size=(32, 32), mode="bilinear", align_corners=False)
        if not torch.isfinite(scores).all():
            raise RuntimeError("Nonfinite E03 semantic scores")
        return scores[:, :1], scores[:, 1:]

    @torch.no_grad()
    def encode_last_block_pair(self, rgb: torch.Tensor, *, mode: str = "both") -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """E04: change only the final patch attention; return full and positive scores."""
        _frozen(self.model)
        if rgb.shape != (1, 3, 512, 512):
            raise ValueError("E04 expects one RGB512 image")
        if mode not in ("both", "value_only", "vv"):
            raise ValueError("Unknown E04 final-block mode")
        visual, geometry = self.model.visual, self._geometry()
        blocks = getattr(visual.transformer, "resblocks", None)
        if blocks is None or len(blocks) != geometry.layers:
            raise RuntimeError(f"E04 requires the {geometry.layers}-block OpenCLIP visual transformer")
        last = blocks[-1]
        attn = last.attn
        head_dim = getattr(attn, "head_dim", geometry.width // attn.num_heads)
        if (attn.in_proj_weight is None or attn.in_proj_bias is None or
            head_dim * attn.num_heads != geometry.width or
            getattr(attn, "logit_scale", None) is not None or
            not isinstance(last.ls_1, nn.Identity) or not isinstance(last.ls_2, nn.Identity) or
            not isinstance(getattr(attn, "ln_inner", nn.Identity()), nn.Identity) or
            getattr(attn, "head_scale", None) is not None):
            raise RuntimeError("Unsupported E04 final-block attention interface")
        x = self._dense_input(rgb)
        hidden = self._embed_tokens(x)
        sequence_first = getattr(self, "_sequence_first", True)
        if sequence_first:
            hidden = hidden.transpose(0, 1)
        for block in blocks[:-1]:
            hidden = block(hidden)
        if sequence_first:
            hidden = hidden.transpose(0, 1)
        patches = hidden[:, 1:]
        normed = last.ln_1(patches)
        w_v = attn.in_proj_weight.chunk(3, dim=0)[2]
        b_v = attn.in_proj_bias.chunk(3)[2]
        value = F.linear(normed, w_v, b_v)
        if mode != "value_only":
            heads = value.reshape(1, geometry.dense_grid ** 2, attn.num_heads, head_dim).transpose(1, 2)
            vv = F.scaled_dot_product_attention(heads, heads, heads, dropout_p=0.0,
                                                scale=head_dim ** -0.5)
            vv = vv.transpose(1, 2).reshape_as(value)
        outputs = {}
        for name in (("value_only", "vv") if mode == "both" else (mode,)):
            attention_value = value if name == "value_only" else vv
            attention_out = attn.out_proj(attention_value)
            attention_out = getattr(last, "ln_attn", nn.Identity())(attention_out)
            tokens = patches + last.ls_1(attention_out)
            tokens = tokens + last.ls_2(last.mlp(last.ln_2(tokens)))
            features = F.normalize((visual.ln_post(tokens) @ visual.proj).float(), dim=-1)
            positive = features @ self.text_positive
            full = positive - (features @ self.text_negative.T).amax(-1)
            scores = torch.stack((full, positive), dim=1).reshape(
                1, 2, geometry.dense_grid, geometry.dense_grid)
            scores = F.interpolate(scores, size=(32, 32), mode="bilinear", align_corners=False)
            if not torch.isfinite(scores).all():
                raise RuntimeError(f"Nonfinite E04 {name} scores")
            outputs[name] = (scores[:, :1], scores[:, 1:])
        return outputs

    @torch.no_grad()
    def encode_window_cls(self, rgb: torch.Tensor) -> torch.Tensor:
        """E08: score 49 fixed 128-pixel crops with the model's native CLS path."""
        _frozen(self.model)
        if rgb.shape != (1, 3, 512, 512):
            raise ValueError("E08 expects one RGB512 image")
        geometry = self._geometry()
        starts = tuple(range(0, 385, 64))
        locations = [(y, x) for y in starts for x in starts]
        total = torch.zeros((1, 1, 32, 32), device=self.device, dtype=torch.float32)
        counts = torch.zeros_like(total)
        source = rgb.to(self.device).float() / 255.0
        for offset in range(0, len(locations), 7):
            chunk = locations[offset:offset + 7]
            crops = torch.cat([source[:, :, y:y + 128, x:x + 128] for y, x in chunk])
            crops = F.interpolate(crops, size=(geometry.native, geometry.native), mode="bilinear", align_corners=False)
            crops = (crops - crops.new_tensor(CLIP_MEAN)[None, :, None, None]) / crops.new_tensor(CLIP_STD)[None, :, None, None]
            features = F.normalize(self.model.encode_image(crops).float(), dim=-1)
            scores = features @ self.text_positive
            for index, (y, x) in enumerate(chunk):
                sy, sx = y // 16, x // 16
                total[0, 0, sy:sy + 8, sx:sx + 8] += scores[index]
                counts[0, 0, sy:sy + 8, sx:sx + 8] += 1
        if not (counts > 0).all():
            raise RuntimeError("E08 CLS windows left uncovered cells")
        result = total / counts
        if not torch.isfinite(result).all():
            raise RuntimeError("Nonfinite E08 CLS score")
        return result


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
    return (GeoRSCLIP(config.georsclip_weights, config.georsclip_sha256, config.device, config.georsclip_model),
            DinoSAT(config.dinov3_source, config.dinov3_weights, config.dinov3_sha256_prefix, config.device))
