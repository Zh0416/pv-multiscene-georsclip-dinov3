"""Validate and read frozen E08 raw maps from the locked L/14 diagnostic."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config
from .encoders import NEGATIVE_CLASSES, POSITIVE_PROMPTS
from .io import read_json, sha256
from .splits import partition_ids


def prepare_e08_source(config: Config, manifest: dict, selected: list[str]) -> dict:
    from .cache import PREPROCESS_VERSION
    from .l14_probe import PROTOCOL
    from .semantic_probe import _check_split

    if (config.semantic_mode != "window_cls_positive_only" or
        config.georsclip_model != "ViT-L-14-336" or
        config.semantic_diagnostic_dir is None or config.semantic_summary_sha256 is None):
        raise ValueError("E08 cache needs a pinned L/14 diagnostic source")
    output = config.output_dir
    manifest_path, split_path = output / "dataset_manifest.json", output / "split_lock.json"
    if read_json(manifest_path) != manifest:
        raise ValueError("E08 cache manifest differs from locked output")
    split = read_json(split_path)
    dev_ids = _check_split(manifest, split)
    if set(selected) != set(dev_ids) or len(selected) != len(dev_ids):
        raise PermissionError("E08 cache requires exactly locked development images; test stays sealed")
    if set(dev_ids) != set(partition_ids(split, "dev")):
        raise ValueError("E08 development IDs differ from locked split")
    source = config.semantic_diagnostic_dir.resolve()
    identity_path, summary_path = source / "identity.json", source / "l14_summary.json"
    if not identity_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Complete L/14 diagnostic directory required; review ZIP alone is insufficient")
    identity, summary = read_json(identity_path), read_json(summary_path)
    identity_sha, summary_sha = sha256(identity_path), sha256(summary_path)
    if (summary_sha != config.semantic_summary_sha256 or
        summary.get("identity_sha256") != identity_sha or
        summary.get("status") != "development_l14_diagnostic_complete" or
        summary.get("development_images") != len(dev_ids) or
        identity.get("protocol") != PROTOCOL or
        identity.get("georsclip_model") != config.georsclip_model or
        identity.get("georsclip_sha256") != config.georsclip_sha256 or
        identity.get("preprocess") != PREPROCESS_VERSION or
        identity.get("augmentation") != config.augmentation or
        identity.get("prompts_positive") != list(POSITIVE_PROMPTS) or
        identity.get("prompts_negative") != list(NEGATIVE_CLASSES) or
        identity.get("manifest_sha256") != sha256(manifest_path) or
        identity.get("split_sha256") != sha256(split_path)):
        raise ValueError("E08 diagnostic identity, summary or locked split differs from approved run")
    code = identity.get("code_sha256", {})
    if any(code.get(name) != sha256(Path(__file__).parent / name)
           for name in ("l14_probe.py", "encoders.py", "semantic_extensions.py", "priors.py", "metrics.py")):
        raise ValueError("E08 source code differs from approved diagnostic; rerun and review before promotion")
    return {"directory": str(source), "identity_sha256": identity_sha,
            "summary_sha256": summary_sha, "open_clip_version": identity["open_clip_version"]}


def read_e08_map(source: Path, image_id: str) -> tuple[np.ndarray, float, str]:
    from .cache import _name

    stem = Path(_name(image_id)).stem
    raw_path, meta_path = source / "raw" / f"{stem}.npz", source / "raw" / f"{stem}.json"
    if not raw_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"E08 raw map missing for {image_id}; rerun diagnose-l14")
    metadata = read_json(meta_path)
    digest = sha256(raw_path)
    if (metadata.get("image_id") != image_id or metadata.get("sha256") != digest or
        metadata.get("native_global_unchanged") is not True):
        raise ValueError(f"E08 raw map integrity mismatch: {image_id}")
    with np.load(raw_path, allow_pickle=False) as archive:
        semantic = archive["window_cls_positive_only"].copy()
        global_raw = float(archive["global_raw"])
    if semantic.shape != (32, 32) or not np.isfinite(semantic).all() or not np.isfinite(global_raw):
        raise ValueError(f"Invalid E08 raw map: {image_id}")
    return semantic.astype(np.float32, copy=False), global_raw, digest
