from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import rasterio
import torch

from .config import Config
from .e08_source import prepare_e08_source, read_e08_map
from .encoders import GEOMETRIES, NEGATIVE_CLASSES, POSITIVE_PROMPTS, load_encoders
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
    rows = {row["id"]: row for row in manifest["images"]}
    selected = sorted(ids if ids is not None else rows)
    if not set(selected) <= rows.keys():
        raise ValueError("Cache contains unknown image ID")
    semantic_source = prepare_e08_source(config, manifest, selected) if config.semantic_mode == "window_cls_positive_only" else None
    geo, dino = load_encoders(config)
    if semantic_source and geo.open_clip_version != semantic_source["open_clip_version"]:
        raise ValueError("OpenCLIP version differs from approved E08 diagnostic")
    folder = config.output_dir / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"schema": "pvstudy-cache-v1", "georsclip_sha256": geo.checkpoint_sha256,
                "open_clip_version": geo.open_clip_version,
                "dinov3_sha256": dino.checkpoint_sha256,
                "dinov3_source": _source_identity(config.dinov3_source, config.dinov3_source_commit),
                "preprocess": PREPROCESS_VERSION, "augmentation": config.augmentation,
                "dino_layers": list(dino.layer_indices),
                "dino_reduction": "get_intermediate_layers(n=4,reshape=True,norm=True), arithmetic mean",
                "georsclip_model": config.georsclip_model,
                "georsclip_dense": geo.geometry.dense_description,
                "prompts_positive": list(POSITIVE_PROMPTS), "prompts_negative": list(NEGATIVE_CLASSES)}
    if semantic_source:
        identity.update({"semantic_mode": config.semantic_mode, "semantic_source": semantic_source})
    index_path = folder / "index.json"
    if index_path.is_file():
        previous = read_json(index_path)
        if previous.get("identity") != identity:
            raise ValueError("Existing feature cache has a different encoder or semantic identity")
        entries = dict(previous["entries"])
    else:
        entries = {}
    for image_id in selected:
        row = rows[image_id]
        image_path = config.image_dir / image_id
        if sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"Image changed after audit: {image_id}")
        path = folder / _name(image_id)
        if image_id in entries:
            entry = entries[image_id]
            if (entry.get("file") != path.name or entry.get("image_sha256") != row["image_sha256"] or
                not path.is_file() or sha256(path) != entry.get("cache_sha256")):
                raise ValueError(f"Existing cache entry integrity mismatch: {image_id}")
            continue
        with rasterio.open(image_path) as image:
            rgb = torch.from_numpy(image.read().copy())[None]
        with torch.no_grad():
            native, global_score = geo.encode(rgb)
            feat = dino.encode(rgb)
        if semantic_source:
            sem, diagnostic_global, raw_sha = read_e08_map(Path(semantic_source["directory"]), image_id)
            if not np.isclose(float(global_score[0]), diagnostic_global, rtol=1e-4, atol=1e-5):
                raise ValueError(f"Native global CLS differs from approved E08 diagnostic: {image_id}")
        else:
            sem, raw_sha = native[0, 0].cpu().numpy().astype(np.float32), None
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, dino=feat[0].cpu().numpy().astype(np.float32),
                            semantic_raw=sem,
                            global_raw=global_score[0].cpu().numpy().astype(np.float32))
        temporary.replace(path)
        entries[image_id] = {"file": path.name, "cache_sha256": sha256(path),
                             "image_sha256": row["image_sha256"]}
        if raw_sha:
            entries[image_id]["semantic_source_sha256"] = raw_sha
        temporary_index = index_path.with_suffix(".tmp.json")
        write_json(temporary_index, {"identity": identity, "entries": entries})
        temporary_index.replace(index_path)
    payload = {"identity": identity, "entries": entries}
    if not index_path.is_file():
        write_json(index_path, payload)
    return payload


def load_feature(config: Config, manifest: dict, index: dict, image_id: str) -> dict:
    row = next(r for r in manifest["images"] if r["id"] == image_id)
    entry = index["entries"][image_id]
    if (index["identity"]["preprocess"] != PREPROCESS_VERSION or
        index["identity"]["augmentation"] != config.augmentation or
        index["identity"]["georsclip_sha256"] != config.georsclip_sha256 or
        index["identity"].get("georsclip_model", "ViT-B/32") != config.georsclip_model or
        index["identity"].get("semantic_mode", "native_full") != config.semantic_mode or
        index["identity"].get("georsclip_dense", GEOMETRIES["ViT-B/32"].dense_description) != GEOMETRIES[config.georsclip_model].dense_description or
        not index["identity"]["dinov3_sha256"].startswith(config.dinov3_sha256_prefix) or
        entry["image_sha256"] != row["image_sha256"]):
        raise ValueError("Cache provenance mismatch")
    if config.semantic_mode == "window_cls_positive_only":
        source = index["identity"].get("semantic_source") or {}
        if (source.get("directory") != str(config.semantic_diagnostic_dir.resolve()) or
            source.get("summary_sha256") != config.semantic_summary_sha256 or
            not entry.get("semantic_source_sha256")):
            raise ValueError("E08 cache source provenance mismatch")
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


def verify_e08_cache(config: Config) -> dict:
    if config.semantic_mode != "window_cls_positive_only":
        raise ValueError("verify-e08-cache requires the E08 configuration")
    manifest = read_json(config.output_dir / "dataset_manifest.json")
    split = read_json(config.output_dir / "split_lock.json")
    from .semantic_probe import _check_split

    ids = _check_split(manifest, split)
    source = prepare_e08_source(config, manifest, ids)
    index = read_cache_index(config)
    if (index["identity"].get("semantic_mode") != config.semantic_mode or
        index["identity"].get("semantic_source") != source or
        set(index["entries"]) != set(ids)):
        raise ValueError("E08 cache index differs from locked diagnostic or development split")
    for image_id in ids:
        entry = index["entries"][image_id]
        raw, diagnostic_global, raw_sha = read_e08_map(Path(source["directory"]), image_id)
        if entry.get("semantic_source_sha256") != raw_sha:
            raise ValueError(f"E08 source digest differs from cache index: {image_id}")
        cached = load_feature(config, manifest, index, image_id)
        if not np.array_equal(cached["semantic_raw"], raw):
            raise ValueError(f"E08 cache map differs from approved raw map: {image_id}")
        if not np.isclose(cached["global_raw"], diagnostic_global, rtol=1e-4, atol=1e-5):
            raise ValueError(f"E08 global CLS differs from approved diagnostic: {image_id}")
    report = {"status": "e08_cache_verified", "development_images": len(ids),
              "cache_index_sha256": sha256(config.output_dir / "cache" / "index.json"),
              "manifest_sha256": sha256(config.output_dir / "dataset_manifest.json"),
              "split_sha256": sha256(config.output_dir / "split_lock.json"),
              "diagnostic_identity_sha256": source["identity_sha256"],
              "diagnostic_summary_sha256": source["summary_sha256"],
              "semantic_mode": config.semantic_mode, "test_images_read": 0}
    write_json(config.output_dir / "e08_cache_verification.json", report)
    return report
