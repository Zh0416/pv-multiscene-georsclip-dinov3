from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    image_dir: Path
    label_dir: Path
    georsclip_weights: Path
    dinov3_weights: Path
    dinov3_source: Path
    output_dir: Path
    device: str = "cuda"
    georsclip_sha256: str = "129bafaa6a097b8be52e2babf27d24f0a934dae919201e538dc698611bd1ea01"
    georsclip_model: str = "ViT-B/32"
    dinov3_sha256_prefix: str = "eadcf0ff"
    dinov3_source_commit: str | None = None
    augmentation: str = "none"

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        base = path.resolve().parent
        for key in ("image_dir", "label_dir", "georsclip_weights", "dinov3_weights", "dinov3_source", "output_dir"):
            if os.name == "nt" and str(raw[key]).startswith("/kaggle/"):
                raise ValueError("Kaggle /kaggle paths require a Linux Kaggle runtime")
            value = Path(raw[key])
            raw[key] = value if value.is_absolute() else (base / value).resolve()
        config = cls(**raw)
        if config.georsclip_model not in ("ViT-B/32", "ViT-L-14-336"):
            raise ValueError("Unsupported GeoRSCLIP architecture")
        if config.augmentation != "none":
            raise ValueError("Only augmentation=none is supported by the frozen-feature cache")
        return config
