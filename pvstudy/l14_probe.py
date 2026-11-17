"""Architecture-isolated development diagnostic for GeoRSCLIP ViT-L/14-336."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import rasterio
import torch

from .cache import PREPROCESS_VERSION
from .config import Config
from .encoders import GeoRSCLIP, NEGATIVE_CLASSES, POSITIVE_PROMPTS
from .io import read_json, sha256
from .priors import area_target
from .semantic_extensions import _previews, evaluate_extension
from .semantic_probe import _atomic_json, _check_split, _checked_input_path, _measure, _runtime

MODES = ["constant", "native_full", "native_positive_only", "sliding_full",
         "sliding_positive_only", "value_only_full", "vv_full",
         "value_only_positive_only", "vv_positive_only", "window_cls_positive_only"]
PROTOCOL = {
    "schema": "pvstudy-l14-semantic-v1", "architecture": "ViT-L-14-336",
    "dense": "RGB512 replicate-pad right/bottom to 518; 37x37 patches; bicubic 24x24 positions; bilinear to 32x32",
    "sliding": "nine native 336 windows, patch-aligned starts 0/98/182; uniform raw-score averaging on 37x37",
    "e04": "last of 24 visual blocks; patch-only value or per-head V-V attention; native global CLS unchanged",
    "e08": "49 source 128 windows at stride 64, bilinear resize to native 336; uniform 8x8 coverage",
    "text": "full positive-minus-hard-negative margin and positive-only cosine scored separately",
    "calibration": "independent site-weighted fit on each of four outer training folds",
    "scope": "locked development only; no decoder training or test access",
    "registration": "new-architecture adaptation; compare with B/32 historical metrics descriptively only",
}


def run_l14_probe(config: Config, *, run_name: str = "l14-semantic-v1") -> dict:
    if config.georsclip_model != "ViT-L-14-336":
        raise ValueError("diagnose-l14 requires georsclip_model=ViT-L-14-336")
    if config.augmentation != "none":
        raise ValueError("The frozen L/14 diagnostic requires augmentation=none")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", run_name):
        raise ValueError("Use a short alphanumeric run name")
    root = config.output_dir
    manifest_path, split_path = root / "dataset_manifest.json", root / "split_lock.json"
    manifest, split = read_json(manifest_path), read_json(split_path)
    ids = _check_split(manifest, split)
    rows = {row["id"]: row for row in manifest["images"]}
    runtime = _runtime(config.device)
    geo = GeoRSCLIP(config.georsclip_weights, config.georsclip_sha256, config.device,
                    config.georsclip_model)
    identity = {
        "protocol": PROTOCOL, "runtime": runtime, "preprocess": PREPROCESS_VERSION,
        "augmentation": config.augmentation, "georsclip_sha256": geo.checkpoint_sha256,
        "georsclip_model": config.georsclip_model, "open_clip_version": geo.open_clip_version,
        "prompts_positive": list(POSITIVE_PROMPTS), "prompts_negative": list(NEGATIVE_CLASSES),
        "manifest_sha256": sha256(manifest_path), "split_sha256": sha256(split_path),
        "code_sha256": {name: sha256(Path(__file__).parent / name) for name in
                        ("l14_probe.py", "encoders.py", "semantic_extensions.py", "priors.py", "metrics.py")},
    }
    folder = root / "diagnostics" / run_name
    identity_path = folder / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("L/14 diagnostic identity changed; use a new --run-name")
    folder.mkdir(parents=True, exist_ok=True)
    _atomic_json(identity_path, identity)
    raw_folder = folder / "raw"
    raw_folder.mkdir(exist_ok=True)
    records, timings = [], []
    for number, image_id in enumerate(ids, 1):
        row = rows[image_id]
        image_path = _checked_input_path(config.image_dir, image_id)
        label_path = _checked_input_path(config.label_dir, image_id)
        if sha256(image_path) != row["image_sha256"] or sha256(label_path) != row["label_sha256"]:
            raise ValueError(f"Input changed since locked audit: {image_id}")
        stem = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:24]
        raw_path, meta_path = raw_folder / f"{stem}.npz", raw_folder / f"{stem}.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            if meta.get("image_id") != image_id or meta.get("sha256") != sha256(raw_path):
                raise ValueError(f"L/14 raw-map integrity mismatch: {image_id}")
            with np.load(raw_path, allow_pickle=False) as archive:
                maps = {mode: archive[mode].copy() for mode in MODES[1:]}
            print(f"[{number}/{len(ids)}] reuse {image_id}", flush=True)
        else:
            with rasterio.open(image_path) as image:
                rgb = torch.from_numpy(image.read().copy())[None]
            maps, seconds, peaks = {}, {}, {}
            native, seconds["native"], peaks["native"] = _measure(
                lambda: geo.encode_semantic_pair(rgb), config.device)
            sliding, seconds["sliding"], peaks["sliding"] = _measure(
                lambda: geo.encode_semantic_pair(rgb, sliding_native=True), config.device)
            for name, pair in (("native", native), ("sliding", sliding)):
                maps[f"{name}_full"] = pair[0][0, 0].cpu().numpy().copy()
                maps[f"{name}_positive_only"] = pair[1][0, 0].cpu().numpy().copy()
            for name in ("value_only", "vv"):
                result, seconds[name], peaks[name] = _measure(
                    lambda name=name: geo.encode_last_block_pair(rgb, mode=name), config.device)
                maps[f"{name}_full"] = result[name][0][0, 0].cpu().numpy().copy()
                maps[f"{name}_positive_only"] = result[name][1][0, 0].cpu().numpy().copy()
            result, seconds["window_cls"], peaks["window_cls"] = _measure(
                lambda: geo.encode_window_cls(rgb), config.device)
            maps["window_cls_positive_only"] = result[0, 0].cpu().numpy().copy()
            check, seconds["native_global_check"], peaks["native_global_check"] = _measure(
                lambda: geo.encode(rgb), config.device)
            if (not np.allclose(check[0][0, 0].cpu().numpy(), maps["native_full"],
                                rtol=1e-4, atol=1e-5) or not torch.isfinite(check[1]).all()):
                raise ValueError(f"Native L/14 full score or global CLS changed: {image_id}")
            temporary = raw_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **maps, global_raw=np.float32(check[1][0].cpu()))
            temporary.replace(raw_path)
            meta = {"image_id": image_id, "sha256": sha256(raw_path), "seconds": seconds,
                    "cuda_peak_allocated_bytes": peaks, "native_global_unchanged": True}
            _atomic_json(meta_path, meta)
            print(f"[{number}/{len(ids)}] extracted {image_id}", flush=True)
        if any(maps[mode].shape != (32, 32) or not np.isfinite(maps[mode]).all()
               for mode in MODES[1:]):
            raise ValueError(f"Invalid L/14 score map: {image_id}")
        with rasterio.open(label_path) as label:
            mask = label.read(1).astype(np.float32)
        records.append({"id": image_id, "site": split["assignment"][image_id]["site"],
                        "mask": mask, "y32": area_target(mask), **maps})
        timings.append(meta)
    result, folds, predictions = evaluate_extension(records, split, MODES)
    result.update({"schema": PROTOCOL["schema"], "status": "development_l14_diagnostic_complete",
                   "protocol": PROTOCOL, "development_images": len(ids),
                   "identity_sha256": sha256(identity_path), "runtime": runtime,
                   "timing_seconds_per_image": {key: float(np.median([t["seconds"][key] for t in timings]))
                                                for key in timings[0]["seconds"]},
                   "limits": "New architecture; old B/32 raw maps and calibrated metrics are not reused. No test evaluation or main-cache replacement."})
    result["previews"] = _previews(config, records, rows, predictions, MODES, folder)
    summary_path = folder / "l14_summary.json"
    _atomic_json(summary_path, result)
    for fold in folds:
        _atomic_json(folder / f"fold-{fold['fold']}.json", fold)
    _atomic_json(folder / "timing.json", {"images": timings})
    bundle = folder / "l14_review.zip"
    temporary = bundle.with_suffix(".tmp.zip")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for path in (summary_path, identity_path, folder / "timing.json",
                     *(folder / f"fold-{i}.json" for i in range(4)),
                     *(folder / item["file"] for item in result["previews"])):
            archive.write(path, path.name)
    temporary.replace(bundle)
    print(f"L/14 diagnostic complete: {summary_path}\nReview bundle: {bundle}", flush=True)
    return result
