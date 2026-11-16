from __future__ import annotations

import json
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
    dinov3_sha256_prefix: str = "eadcf0ff"
    dinov3_source_commit: str | None = None
    augmentation: str = "none"

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        base = path.resolve().parent
        for key in ("image_dir", "label_dir", "georsclip_weights", "dinov3_weights", "dinov3_source", "output_dir"):
            value = Path(raw[key])
            raw[key] = value if value.is_absolute() else (base / value).resolve()
        config = cls(**raw)
        if config.augmentation != "none":
            raise ValueError("Only augmentation=none is supported by the frozen-feature cache")
        return config
