"""Isolated E05 development diagnostic; never trains or replaces the main pipeline."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw

from .cache import PREPROCESS_VERSION, read_cache_index
from .config import Config
from .encoders import GeoRSCLIP, NEGATIVE_CLASSES, POSITIVE_PROMPTS
from .io import read_json, sha256, write_json
from .metrics import area_calibration, paired_site_bootstrap, site_metrics
from .priors import area_target, fit_patch, site_weights
from .splits import partition_ids

PROTOCOL = {
    "schema": "pvstudy-e05-v1", "input_size": 512, "window_size": 224,
    "starts": [0, 96, 192, 288], "window_batch": 4,
    "stitch": "mean raw cosine margins on aligned 16x16 patch grid, then bilinear to 32x32",
    "align_corners": False, "warmup_pairs": 2, "timing_repeats": 3,
    "timing_scope": "encoder call including normalization/device transfer/stitch; excludes disk IO and weight loading",
    "calibration": "separate site-weighted fit on each outer training fold; no decoder training",
    "preview_selection": "smallest/median/largest positive target fraction and first negative ID, development only",
}


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    write_json(temporary, payload)
    temporary.replace(path)


def _checked_input_path(root: Path, image_id: str) -> Path:
    path = (root / image_id).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("Image ID escapes configured data directory")
    return path


def _check_split(manifest: dict, split: dict) -> list[str]:
    if not manifest.get("manual_qc_confirmed") or split.get("status") != "locked_after_manual_qc":
        raise PermissionError("E05 requires the existing manually locked split")
    if not split.get("qc_sha256") or split["qc_sha256"] != manifest.get("manual_qc_sha256"):
        raise ValueError("Manifest and split QC identities differ")
    ids = partition_ids(split, "dev")
    if not ids or set(split["assignment"]) != {r["id"] for r in manifest["images"]}:
        raise ValueError("Manifest and split IDs differ")
    if len({r["id"] for r in manifest["images"]}) != len(manifest["images"]):
        raise ValueError("Manifest contains duplicate image IDs")
    sites = {}
    for info in split["assignment"].values():
        if info["partition"] not in ("dev", "test") or (info["partition"] == "dev" and info["fold"] not in range(4)) or (info["partition"] == "test" and info["fold"] is not None):
            raise ValueError("Invalid locked split assignment")
        group = (info["partition"], info["fold"])
        if info["site"] in sites and sites[info["site"]] != group:
            raise ValueError("Site crosses partitions or development folds")
        sites[info["site"]] = group
    for fold in range(4):
        if not partition_ids(split, "train", fold) or not partition_ids(split, "validation", fold):
            raise ValueError("E05 needs all four development folds")
    return ids


def _runtime(device: str) -> dict:
    is_cuda = torch.device(device).type == "cuda"
    if is_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return {"device": device, "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if is_cuda else None,
            "torch_cpu_threads": torch.get_num_threads()}


def _measure(function, device: str):
    is_cuda = torch.device(device).type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = function()
    if is_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) if is_cuda else None
    return result, elapsed, peak


def _cached_native(config: Config, row: dict, entry: dict) -> tuple[np.ndarray, float]:
    if entry["image_sha256"] != row["image_sha256"]:
        raise ValueError("Feature cache and manifest image hashes differ")
    folder = config.output_dir / "cache"
    path = folder / entry["file"]
    if path.resolve().parent != folder.resolve() or sha256(path) != entry["cache_sha256"]:
        raise ValueError("Native feature cache file integrity mismatch")
    with np.load(path, allow_pickle=False) as archive:
        raw = archive["semantic_raw"].copy()
        global_score = float(archive["global_raw"])
    if raw.shape != (32, 32) or not np.isfinite(raw).all() or not np.isfinite(global_score):
        raise ValueError("Native cached semantic score layout/value differs")
    return raw, global_score


def evaluate_folds(records: list[dict], split: dict) -> tuple[dict, list[dict], dict]:
    """Calibration sees outer training sites only; every dev site is evaluated once."""
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids) or set(ids) != set(partition_ids(split, "dev")):
        raise ValueError("E05 evaluation requires exactly the development records")
    by_id = {r["id"]: r for r in records}
    for r in records:
        if r["site"] != split["assignment"][r["id"]]["site"]:
            raise ValueError("Record site differs from locked split")
    folds, all_rows = [], {key: [] for key in ("constant", "native512", "sliding224")}
    predictions = {key: {} for key in all_rows}
    for fold in range(4):
        train = [by_id[i] for i in partition_ids(split, "train", fold)]
        validation = [by_id[i] for i in partition_ids(split, "validation", fold)]
        prevalence = float(np.sum([r["y32"].mean() * w for r, w in zip(train, site_weights(train, pixels=False))]))
        metrics, calibrators = {}, {}
        for mode in all_rows:
            if mode == "constant":
                scores = {r["id"]: np.full((32, 32), prevalence, np.float32) for r in validation}
            else:
                calibrator = fit_patch(train, mode)
                calibrators[mode] = calibrator.as_dict()
                scores = {r["id"]: calibrator.predict(r[mode]) for r in validation}
            metrics[mode] = {**site_metrics(validation, scores, cam=True), **area_calibration(validation, scores)}
            all_rows[mode].extend(metrics[mode]["site_rows"])
            predictions[mode].update(scores)
        folds.append({"fold": fold, "train_ids": [r["id"] for r in train],
                      "validation_ids": [r["id"] for r in validation],
                      "calibrators": calibrators, "metrics": metrics})
    aggregate = {}
    for mode, rows in all_rows.items():
        if len({r["site"] for r in rows}) != len(rows):
            raise ValueError("Site appears in multiple outer validation folds")
        aggregate[mode] = {key: float(np.mean([r[key] for r in rows if r[key] is not None]))
                           for key in ("ap", "dice", "iou", "negative_fp")}
        if not all(np.isfinite(v) for v in aggregate[mode].values()):
            raise ValueError("E05 requires positive and negative development sites")
    left = {r["site"]: r["ap"] for r in all_rows["sliding224"] if r["ap"] is not None}
    right = {r["site"]: r["ap"] for r in all_rows["native512"] if r["ap"] is not None}
    comparison = paired_site_bootstrap(left, right)
    comparison["negative_fp_delta"] = aggregate["sliding224"]["negative_fp"] - aggregate["native512"]["negative_fp"]
    comparison["fold_ap_deltas"] = [f["metrics"]["sliding224"]["site_macro_ap"] - f["metrics"]["native512"]["site_macro_ap"] for f in folds]
    e01 = {}
    for mode in ("native512", "sliding224"):
        delta = aggregate[mode]["ap"] - aggregate["constant"]["ap"]
        positive_folds = sum(f["metrics"][mode]["site_macro_ap"] > f["metrics"]["constant"]["site_macro_ap"] for f in folds)
        e01[mode] = {"ap_delta_vs_constant": delta, "positive_folds": positive_folds,
                     "quality_pass": delta >= 0.05 and positive_folds >= 3}
    return {"aggregate": aggregate, "sliding_minus_native": comparison, "E01_semantic": e01}, folds, predictions


def _previews(config: Config, records: list[dict], manifest_rows: dict, predictions: dict, folder: Path) -> list[dict]:
    positive = sorted([r for r in records if r["mask"].any()], key=lambda r: (manifest_rows[r["id"]]["target_fraction"], r["id"]))
    negative = sorted([r for r in records if not r["mask"].any()], key=lambda r: r["id"])
    chosen = [("smallest_positive", positive[0]), ("median_positive", positive[len(positive)//2]),
              ("largest_positive", positive[-1]), ("first_negative", negative[0])]
    exported = []
    for reason, r in chosen:
        with rasterio.open(_checked_input_path(config.image_dir, r["id"])) as image:
            rgb = image.read().transpose(1, 2, 0)
        panels = [Image.fromarray(rgb), Image.fromarray((r["mask"] * 255).astype(np.uint8)).convert("RGB")]
        for mode in ("native512", "sliding224"):
            p = np.clip(predictions[mode][r["id"]], 0, 1)
            color = np.stack((p, np.zeros_like(p), 1-p), axis=-1)
            panels.append(Image.fromarray(np.round(color*255).astype(np.uint8)).resize((512,512), Image.Resampling.NEAREST))
        canvas = Image.new("RGB", (1024, 1080), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (panel, label) in enumerate(zip(panels, ("RGB", "Ground truth", "Native512 area probability", "Sliding224 area probability"))):
            x, y = (i % 2)*512, (i // 2)*540
            draw.text((x+8,y+6),label,fill="black")
            canvas.paste(panel, (x,y+26))
        path = folder / f"preview-{reason}.png"
        canvas.save(path)
        exported.append({"selection_rule": reason, "image_id": r["id"], "file": path.name,
                         "color_scale": "fixed 0 blue to 1 red, no per-image min-max"})
    return exported


def run_semantic_probe(config: Config, *, run_name: str = "e05-native224-v1", recompute_native: bool = False) -> dict:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", run_name):
        raise ValueError("Use a short alphanumeric run name")
    root = config.output_dir
    manifest = read_json(root / "dataset_manifest.json")
    split = read_json(root / "split_lock.json")
    ids = _check_split(manifest, split)
    index = read_cache_index(config)
    cache_identity = index["identity"]
    if not set(ids) <= set(index["entries"]):
        raise ValueError("Development feature cache is incomplete; restore the original output directory")
    expected = {"preprocess": PREPROCESS_VERSION, "augmentation": "none", "georsclip_sha256": config.georsclip_sha256,
                "prompts_positive": list(POSITIVE_PROMPTS), "prompts_negative": list(NEGATIVE_CLASSES),
                "georsclip_dense": "native patch tokens, bicubic 7x7 positional interpolation to 16x16"}
    if config.augmentation != "none" or any(cache_identity.get(k) != v for k,v in expected.items()):
        raise ValueError("Existing cache does not match the registered native semantic baseline")
    references = {str(f): root / f"fold-{f}/seed-42/result-cam.json" for f in range(4)}
    if recompute_native and not all(p.is_file() for p in references.values()):
        raise ValueError("Recomputing native scores requires all four original seed-42 CAM reports; restore the E05 recovery bundle")
    baseline_source = "recomputed_from_checkpoint_verified_against_original_C1_metrics" if recompute_native else "original_feature_cache"
    runtime = _runtime(config.device)
    print(f"E05: {len(ids)} development images; loading GeoRSCLIP only", flush=True)
    geo = GeoRSCLIP(config.georsclip_weights, config.georsclip_sha256, config.device)
    if geo.open_clip_version != cache_identity.get("open_clip_version"):
        raise ValueError("OpenCLIP version differs from the original cache; restore that version first")
    rows = {r["id"]: r for r in manifest["images"]}
    identity = {"protocol": PROTOCOL, "runtime": runtime, "open_clip_version": geo.open_clip_version,
        "native_baseline_source": baseline_source,
        "original_C1_report_sha256": {f: sha256(p) for f,p in references.items() if p.is_file()},
        "georsclip_sha256": geo.checkpoint_sha256, "cache_identity": cache_identity,
        "manifest_sha256": sha256(root / "dataset_manifest.json"), "split_sha256": sha256(root / "split_lock.json"),
        "native_entries": {i: index["entries"][i] for i in ids},
        "code_sha256": {name: sha256(Path(__file__).parent/name) for name in ("semantic_probe.py", "encoders.py", "priors.py", "metrics.py")}}
    folder = root / "diagnostics" / run_name
    folder.mkdir(parents=True, exist_ok=True)
    identity_path = folder / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Diagnostic identity changed (code/runtime/data); keep this run and use a new --run-name")
    _atomic_json(identity_path, identity)
    raw_folder = folder / "raw"
    raw_folder.mkdir(exist_ok=True)
    records, timings, warmed = [], [], False
    for number, image_id in enumerate(ids, 1):
        row = rows[image_id]
        image_path = _checked_input_path(config.image_dir, image_id)
        label_path = _checked_input_path(config.label_dir, image_id)
        if sha256(image_path) != row["image_sha256"] or sha256(label_path) != row["label_sha256"]:
            raise ValueError(f"Input changed after audit: {image_id}")
        if index["entries"][image_id]["image_sha256"] != row["image_sha256"]:
            raise ValueError("Original cache index and manifest image hashes differ")
        native, native_global = (None, None) if recompute_native else _cached_native(config, row, index["entries"][image_id])
        stem = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:24]
        raw_path, metadata_path = raw_folder / f"{stem}.npz", raw_folder / f"{stem}.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            if metadata["image_id"] != image_id or sha256(raw_path) != metadata["sha256"]:
                raise ValueError("Diagnostic raw-map integrity mismatch")
            with np.load(raw_path, allow_pickle=False) as archive:
                sliding = archive["sliding224"].copy()
                if recompute_native:
                    native = archive["native512"].copy()
                    native_global = float(archive["global_raw"])
            print(f"[{number}/{len(ids)}] reuse {image_id}", flush=True)
        else:
            with rasterio.open(image_path) as image:
                rgb = torch.from_numpy(image.read().copy())[None]
            if not warmed:
                for _ in range(PROTOCOL["warmup_pairs"]):
                    geo.encode(rgb)
                    geo.encode_sliding224(rgb)
                warmed = True
            durations, peaks = {m: [] for m in ("native512", "sliding224")}, {}
            for repeat in range(PROTOCOL["timing_repeats"]):
                order = ("native512", "sliding224") if (number+repeat)%2 else ("sliding224", "native512")
                for mode in order:
                    function = (lambda: geo.encode(rgb)) if mode == "native512" else (lambda: geo.encode_sliding224(rgb))
                    value, elapsed, peak = _measure(function, config.device)
                    durations[mode].append(elapsed)
                    peaks[mode] = max(peaks.get(mode) or 0, peak or 0) if peak is not None else None
                    if mode == "native512":
                        fresh, global_score = value
                        if native is None:
                            native = fresh[0,0].cpu().numpy().copy()
                            native_global = float(global_score[0])
                        elif not np.allclose(fresh[0,0].cpu().numpy(), native, rtol=1e-4, atol=1e-5) or not np.isclose(float(global_score[0]), native_global, rtol=1e-4, atol=1e-5):
                            origin = "repeat inference" if recompute_native else "original cache"
                            raise ValueError(f"Fresh native/CLS output differs from {origin}; investigate baseline before E05")
                    else:
                        sliding = value[0,0].cpu().numpy().copy()
            temporary = raw_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, sliding224=sliding, native512=native, global_raw=np.float32(native_global))
            temporary.replace(raw_path)
            metadata = {"image_id": image_id, "sha256": sha256(raw_path), "seconds": durations,
                        "cuda_peak_allocated_bytes": peaks, "native_baseline_source": baseline_source,
                        "fresh_native_and_global_match_original_cache": None if recompute_native else True}
            _atomic_json(metadata_path, metadata)
            print(f"[{number}/{len(ids)}] extracted {image_id}", flush=True)
        if sliding.shape != (32,32) or not np.isfinite(sliding).all():
            raise ValueError("Invalid sliding score map")
        if native.shape != (32,32) or not np.isfinite(native).all() or not np.isfinite(native_global):
            raise ValueError("Invalid native score map")
        with rasterio.open(label_path) as label:
            mask = label.read(1).astype(np.float32)
        records.append({"id": image_id, "site": split["assignment"][image_id]["site"], "mask": mask,
                        "y32": area_target(mask), "native512": native, "sliding224": sliding})
        timings.append(metadata)
    summary, folds, predictions = evaluate_folds(records, split)
    baseline_checks = []
    for fold in folds:
        prior_result = root / f"fold-{fold['fold']}" / "seed-42" / "result-cam.json"
        if prior_result.exists():
            old = read_json(prior_result)
            if old["validation_ids"] != fold["validation_ids"] or old["train_ids"] != fold["train_ids"]:
                raise ValueError("Earlier native CAM uses different split IDs")
            previous = {r["site"]:r for r in old["cam"]["C1"]["site_rows"]}
            current = {r["site"]:r for r in fold["metrics"]["native512"]["site_rows"]}
            if previous.keys() != current.keys():
                raise ValueError("Earlier native CAM uses different sites")
            for site in current:
                for key in ("ap","dice","iou","negative_fp"):
                    a,b = current[site][key],previous[site][key]
                    if (a is None) != (b is None) or (a is not None and not np.isclose(a,b,rtol=0,atol=1e-6)):
                        raise ValueError("Recomputed C1 differs from earlier CAM result")
            baseline_checks.append(fold["fold"])
        _atomic_json(folder / f"fold-{fold['fold']}.json", fold)
    if recompute_native and baseline_checks != list(range(4)):
        raise ValueError("Recovery requires reproducing all four original C1 reports")
    medians = {m: [float(np.median(t["seconds"][m])) for t in timings] for m in ("native512", "sliding224")}
    ratio = float(np.mean(medians["sliding224"]) / np.mean(medians["native512"]))
    ratios = [b/a for a,b in zip(medians["native512"],medians["sliding224"])]
    delta = summary["sliding_minus_native"]["mean_delta"]
    fp_delta = summary["sliding_minus_native"]["negative_fp_delta"]
    summary.update({"schema": PROTOCOL["schema"], "status": "development_semantic_probe_complete",
        "protocol": PROTOCOL, "runtime": runtime, "identity_sha256": sha256(identity_path),
        "development_images": len(ids), "earlier_C1_reproduced_folds": baseline_checks,
        "native_baseline_source": baseline_source,
        "original_native_arrays_verified": not recompute_native,
        "timing": {"mean_seconds_per_image": {m:float(np.mean(v)) for m,v in medians.items()},
                   "sliding_to_native_ratio_of_means": ratio,
                   "per_image_ratio_p50_p95_max": [float(v) for v in np.quantile(ratios,[0.5,0.95,1])]},
        "E05_replacement": {"ap_gain_at_least_0_02": delta >= 0.02, "latency_ratio_at_most_4": ratio <= 4,
            "gpu_latency_measured": runtime["gpu"] is not None,
            "negative_fp_delta_at_most_0_005": fp_delta <= 0.005,
            "positive_folds": sum(v>0 for v in summary["sliding_minus_native"]["fold_ap_deltas"]),
            "quality_pass": runtime["gpu"] is not None and delta >= 0.02 and ratio <= 4 and fp_delta <= 0.005 and
                            sum(v>0 for v in summary["sliding_minus_native"]["fold_ap_deltas"]) >= 3},
        "limits": "Single-route E05 development diagnostic, not a G2 bank/seed check or final pipeline selection. Bootstrap is descriptive. GPU latency must be measured on target hardware. No main cache, CAM or decoder result is replaced."})
    summary["previews"] = _previews(config, records, rows, predictions, folder)
    _atomic_json(folder / "timing.json", {"images": timings})
    _atomic_json(folder / "e05_summary.json", summary)
    for mode in ("native512", "sliding224"):
        target = folder / "predictions" / mode
        target.mkdir(parents=True, exist_ok=True)
        for image_id, score in predictions[mode].items():
            name = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:24]
            np.savez_compressed(target / f"{name}.npz", probability=score)
    bundle = folder / "e05_review.zip"
    with ZipFile(bundle.with_suffix(".tmp.zip"), "w", ZIP_DEFLATED) as archive:
        for path in sorted(folder.glob("*.json")) + sorted(folder.glob("preview-*.png")):
            archive.write(path, path.name)
    bundle.with_suffix(".tmp.zip").replace(bundle)
    print(f"E05 complete: {folder / 'e05_summary.json'}\nReview bundle: {bundle}", flush=True)
    return summary
