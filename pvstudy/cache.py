from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import rasterio
import torch

from .config import Config
from .encoders import NEGATIVE_CLASSES, POSITIVE_PROMPTS, load_encoders
from .io import read_json, sha256, write_json

PREPROCESS_VERSION = "rgb-uint8-clip-openai-sat493m-v1"


def _name(image_id: str) -> str:
    return hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:24] + ".npz"


def _source_identity(source: Path, declared_commit: str | None) -> dict:
    result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True)
    root = subprocess.run(["git", "-C", str(source), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    actual = result.stdout.strip() if result.returncode == 0 and root.returncode == 0 and Path(root.stdout.strip()).resolve() == source.resolve() else None
    if actual and declared_commit and actual != declared_commit:
        raise ValueError("DINOv3 declared source commit differs from Git HEAD")
    if not actual and not declared_commit:
        raise RuntimeError("A non-Git DINOv3 source needs dinov3_source_commit in config")
    digest = hashlib.sha256()
    files = sorted((source / "dinov3").rglob("*.py"))
    if not files:
        raise FileNotFoundError("DINOv3 source tree has no Python files")
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return {"commit": actual or declared_commit, "commit_verified_by_git": bool(actual),
            "python_tree_sha256": digest.hexdigest(), "python_file_count": len(files)}


def build_cache(config: Config, manifest: dict, ids: list[str] | None = None) -> dict:
    geo, dino = load_encoders(config)
    rows = {row["id"]: row for row in manifest["images"]}
    selected = sorted(ids if ids is not None else rows)
    if not set(selected) <= rows.keys():
        raise ValueError("Cache contains unknown image ID")
    folder = config.output_dir / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"schema": "pvstudy-cache-v1", "georsclip_sha256": geo.checkpoint_sha256,
                "open_clip_version": geo.open_clip_version,
                "dinov3_sha256": dino.checkpoint_sha256,
                "dinov3_source": _source_identity(config.dinov3_source, config.dinov3_source_commit),
                "preprocess": PREPROCESS_VERSION, "augmentation": config.augmentation,
                "dino_layers": list(dino.layer_indices),
                "dino_reduction": "get_intermediate_layers(n=4,reshape=True,norm=True), arithmetic mean",
                "georsclip_dense": "native patch tokens, bicubic 7x7 positional interpolation to 16x16",
                "prompts_positive": POSITIVE_PROMPTS, "prompts_negative": NEGATIVE_CLASSES}
    entries = {}
    for image_id in selected:
        row = rows[image_id]
        image_path = config.image_dir / image_id
        if sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"Image changed after audit: {image_id}")
        with rasterio.open(image_path) as image:
            rgb = torch.from_numpy(image.read().copy())[None]
        with torch.no_grad():
            sem, global_score = geo.encode(rgb)
            feat = dino.encode(rgb)
        path = folder / _name(image_id)
        np.savez_compressed(path, dino=feat[0].cpu().numpy().astype(np.float32),
                            semantic_raw=sem[0, 0].cpu().numpy().astype(np.float32),
                            global_raw=global_score[0].cpu().numpy().astype(np.float32))
        entries[image_id] = {"file": path.name, "cache_sha256": sha256(path),
                             "image_sha256": row["image_sha256"]}
    payload = {"identity": identity, "entries": entries}
    write_json(folder / "index.json", payload)
    return payload


def load_feature(config: Config, manifest: dict, index: dict, image_id: str) -> dict:
    row = next(r for r in manifest["images"] if r["id"] == image_id)
    entry = index["entries"][image_id]
    if (index["identity"]["preprocess"] != PREPROCESS_VERSION or
        index["identity"]["augmentation"] != config.augmentation or
        index["identity"]["georsclip_sha256"] != config.georsclip_sha256 or
        not index["identity"]["dinov3_sha256"].startswith(config.dinov3_sha256_prefix) or
        entry["image_sha256"] != row["image_sha256"]):
        raise ValueError("Cache provenance mismatch")
    path = config.output_dir / "cache" / entry["file"]
    if path.resolve().parent != (config.output_dir / "cache").resolve() or sha256(path) != entry["cache_sha256"]:
        raise ValueError("Cache file integrity mismatch")
    with np.load(path, allow_pickle=False) as archive:
        feat = archive["dino"].copy()
        sem = archive["semantic_raw"].copy()
        global_score = float(archive["global_raw"])
    if feat.shape != (1024, 32, 32) or sem.shape != (32, 32) or not np.isfinite(feat).all() or not np.isfinite(sem).all() or not np.isfinite(global_score):
        raise ValueError("Invalid cached feature layout or value")
    return {"dino": feat, "semantic_raw": sem, "global_raw": global_score}


def load_record(config: Config, manifest: dict, index: dict, split: dict, image_id: str, *, allow_test: bool = False) -> dict:
    info = split["assignment"][image_id]
    if info["partition"] == "test" and not (allow_test and split["status"] == "locked_after_manual_qc"):
        raise PermissionError("Test record is sealed")
    row = next(r for r in manifest["images"] if r["id"] == image_id)
    if sha256(config.label_dir / image_id) != row["label_sha256"]:
        raise ValueError(f"Label changed after audit: {image_id}")
    with rasterio.open(config.label_dir / image_id) as label, rasterio.open(config.image_dir / image_id) as image:
        mask = label.read(1).astype(np.float32)
        rgb = image.read().copy()
    item = load_feature(config, manifest, index, image_id)
    item.update({"id": image_id, "site": info["site"], "mask": mask, "rgb": rgb})
    return item


def read_cache_index(config: Config) -> dict:
    return read_json(config.output_dir / "cache" / "index.json")
