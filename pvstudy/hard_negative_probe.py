"""E03: isolate the contribution of hard-negative text on locked development sites."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw

from .cache import PREPROCESS_VERSION, read_cache_index
from .config import Config
from .encoders import GeoRSCLIP, NEGATIVE_CLASSES, POSITIVE_PROMPTS
from .io import read_json, sha256
from .metrics import area_calibration, paired_site_bootstrap, site_metrics
from .priors import area_target, fit_patch, site_weights
from .semantic_probe import _atomic_json, _check_split, _checked_input_path, _measure, _runtime
from .splits import partition_ids

MODES = ("constant", "native_full", "native_positive_only", "sliding_full", "sliding_positive_only")
PROTOCOL = {
    "schema": "pvstudy-e03-v1",
    "primary_comparison": "native_full versus native_positive_only",
    "secondary_comparison": "sliding_full versus sliding_positive_only after E05",
    "positive_only_raw_score": "cosine(normalized visual patch, normalized mean positive text vector)",
    "full_raw_score": "positive_only_raw_score minus max hard-negative cosine",
    "calibration": "independent site-weighted calibrator for each score path on each outer training fold",
    "fixed_cam_threshold": 0.25,
    "hard_negative_retention": "FP decrease >=0.005 and positive-site AP decrease <=0.01",
    "preview_selection": "smallest/median/largest positive target fraction and first negative ID, development only",
}


def _same_metrics(left: dict, right: dict, *, context: str) -> None:
    if set(left) != set(right):
        raise ValueError(f"{context}: different validation sites")
    for site in left:
        for key in ("ap", "dice", "iou", "negative_fp"):
            a, b = left[site][key], right[site][key]
            if (a is None) != (b is None) or (a is not None and not np.isclose(a, b, rtol=0, atol=1e-6)):
                raise ValueError(f"{context}: original metric differs at {site}/{key}")


def evaluate_e03(records: list[dict], split: dict) -> tuple[dict, list[dict], dict]:
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(partition_ids(split, "dev")):
        raise ValueError("E03 evaluation needs exactly the locked development records")
    by_id = {r["id"]: r for r in records}
    if any(r["site"] != split["assignment"][r["id"]]["site"] for r in records):
        raise ValueError("E03 record site differs from locked split")
    folds, rows = [], {mode: [] for mode in MODES}
    predictions = {mode: {} for mode in MODES}
    for fold in range(4):
        train = [by_id[i] for i in partition_ids(split, "train", fold)]
        validation = [by_id[i] for i in partition_ids(split, "validation", fold)]
        prevalence = float(sum(r["y32"].mean() * w for r, w in zip(train, site_weights(train, pixels=False))))
        metrics, calibrators = {}, {}
        for mode in MODES:
            if mode == "constant":
                scores = {r["id"]: np.full((32, 32), prevalence, np.float32) for r in validation}
            else:
                calibrator = fit_patch(train, mode)
                calibrators[mode] = calibrator.as_dict()
                scores = {r["id"]: calibrator.predict(r[mode]) for r in validation}
            metrics[mode] = {**site_metrics(validation, scores, cam=True), **area_calibration(validation, scores)}
            rows[mode].extend(metrics[mode]["site_rows"])
            predictions[mode].update(scores)
        folds.append({"fold": fold, "train_ids": [r["id"] for r in train],
                      "validation_ids": [r["id"] for r in validation],
                      "calibrators": calibrators, "metrics": metrics})
    aggregates = {}
    for mode, site_rows in rows.items():
        if len(site_rows) != len({r["site"] for r in site_rows}):
            raise ValueError("E03 found a site in more than one validation fold")
        aggregates[mode] = {key: float(np.mean([r[key] for r in site_rows if r[key] is not None]))
                            for key in ("ap", "dice", "iou", "negative_fp")}
        if not all(np.isfinite(value) for value in aggregates[mode].values()):
            raise ValueError("E03 needs both positive and negative development sites")
    comparisons = {}
    for geometry in ("native", "sliding"):
        full, positive = f"{geometry}_full", f"{geometry}_positive_only"
        full_ap = {r["site"]: r["ap"] for r in rows[full] if r["ap"] is not None}
        positive_ap = {r["site"]: r["ap"] for r in rows[positive] if r["ap"] is not None}
        full_fp = {r["site"]: r["negative_fp"] for r in rows[full] if r["negative_fp"] is not None}
        positive_fp = {r["site"]: r["negative_fp"] for r in rows[positive] if r["negative_fp"] is not None}
        ap = paired_site_bootstrap(full_ap, positive_ap)
        fp = paired_site_bootstrap(full_fp, positive_fp)
        fp_reduction = -fp["mean_delta"]
        ap_drop = -ap["mean_delta"]
        comparisons[geometry] = {
            "full_minus_positive_only_ap": ap,
            "full_minus_positive_only_negative_fp": fp,
            "negative_fp_reduction_from_hard_negatives": fp_reduction,
            "ap_decrease_from_hard_negatives": ap_drop,
            "fold_ap_deltas": [f["metrics"][full]["site_macro_ap"] - f["metrics"][positive]["site_macro_ap"] for f in folds],
            "fold_negative_fp_deltas": [f["metrics"][full]["negative_site_fp"] - f["metrics"][positive]["negative_site_fp"] for f in folds],
            "retain_hard_negatives": fp_reduction >= 0.005 and ap_drop <= 0.01,
            "positive_only_E01": {
                "ap_delta_vs_constant": aggregates[positive]["ap"] - aggregates["constant"]["ap"],
                "positive_folds": sum(f["metrics"][positive]["site_macro_ap"] > f["metrics"]["constant"]["site_macro_ap"] for f in folds),
            },
        }
        e01 = comparisons[geometry]["positive_only_E01"]
        e01["quality_pass"] = e01["ap_delta_vs_constant"] >= 0.05 and e01["positive_folds"] >= 3
    return {"aggregate": aggregates, "E03": comparisons}, folds, predictions


def _preview(config: Config, records: list[dict], rows: dict, predictions: dict, folder: Path) -> list[dict]:
    positive = sorted([r for r in records if r["mask"].any()], key=lambda r: (rows[r["id"]]["target_fraction"], r["id"]))
    negative = sorted([r for r in records if not r["mask"].any()], key=lambda r: r["id"])
    selected = (("smallest_positive", positive[0]), ("median_positive", positive[len(positive)//2]),
                ("largest_positive", positive[-1]), ("first_negative", negative[0]))
    exported = []
    for reason, record in selected:
        with rasterio.open(_checked_input_path(config.image_dir, record["id"])) as image:
            rgb = Image.fromarray(image.read().transpose(1, 2, 0))
        panels = [(rgb, "RGB"), (Image.fromarray((record["mask"]*255).astype(np.uint8)).convert("RGB"), "Ground truth")]
        for mode in MODES[1:]:
            p = np.clip(predictions[mode][record["id"]], 0, 1)
            color = np.stack((p, np.zeros_like(p), 1-p), axis=-1)
            panels.append((Image.fromarray(np.round(color*255).astype(np.uint8)).resize((512,512), Image.Resampling.NEAREST), mode))
        canvas = Image.new("RGB", (1536, 1080), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (panel, label) in enumerate(panels):
            x, y = (i % 3)*512, (i // 3)*540
            draw.text((x+8,y+6), label, fill="black")
            canvas.paste(panel, (x,y+26))
        path = folder / f"preview-{reason}.png"
        canvas.save(path)
        exported.append({"selection_rule": reason, "image_id": record["id"], "file": path.name,
                         "color_scale": "fixed 0 blue to 1 red, no per-image min-max"})
    return exported


def run_hard_negative_probe(config: Config, *, e05_run_name: str = "e05-recovered-native-v1",
                            run_name: str = "e03-no-hard-negatives-v1") -> dict:
    for name in (e05_run_name, run_name):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name):
            raise ValueError("Diagnostic run names must be short and alphanumeric")
    if run_name == e05_run_name:
        raise ValueError("E03 output directory must differ from E05")
    root = config.output_dir
    manifest = read_json(root / "dataset_manifest.json")
    split = read_json(root / "split_lock.json")
    ids = _check_split(manifest, split)
    index = read_cache_index(config)
    e05 = root / "diagnostics" / e05_run_name
    e05_identity_path, e05_summary_path = e05 / "identity.json", e05 / "e05_summary.json"
    if not e05_identity_path.is_file() or not e05_summary_path.is_file():
        raise FileNotFoundError("E03 requires the complete E05 diagnostic directory with raw maps; rerun diagnose-semantic if the session reset")
    e05_identity, e05_summary = read_json(e05_identity_path), read_json(e05_summary_path)
    if e05_summary.get("status") != "development_semantic_probe_complete" or e05_summary.get("development_images") != len(ids):
        raise ValueError("E05 reference run is incomplete")
    if e05_summary.get("identity_sha256") != sha256(e05_identity_path):
        raise ValueError("E05 identity digest differs from summary")
    cache_identity = index["identity"]
    expected = {"preprocess": PREPROCESS_VERSION, "augmentation": "none", "georsclip_sha256": config.georsclip_sha256,
                "prompts_positive": list(POSITIVE_PROMPTS), "prompts_negative": list(NEGATIVE_CLASSES),
                "georsclip_dense": "native patch tokens, bicubic 7x7 positional interpolation to 16x16"}
    if config.augmentation != "none" or any(cache_identity.get(k) != v for k, v in expected.items()):
        raise ValueError("Original cache identity differs from the registered GeoRSCLIP baseline")
    if (e05_identity.get("cache_identity") != cache_identity or
        e05_identity.get("manifest_sha256") != sha256(root / "dataset_manifest.json") or
        e05_identity.get("split_sha256") != sha256(root / "split_lock.json") or
        e05_identity.get("georsclip_sha256") != config.georsclip_sha256 or
        e05_identity.get("open_clip_version") != cache_identity.get("open_clip_version") or
        e05_summary.get("earlier_C1_reproduced_folds") != list(range(4))):
        raise ValueError("E05 references do not match the locked original experiment")
    rows = {r["id"]: r for r in manifest["images"]}
    if not set(ids) <= set(index["entries"]) or e05_identity.get("native_entries") != {i: index["entries"][i] for i in ids}:
        raise ValueError("E05 and original development cache indices differ")
    references = {str(f): root / f"fold-{f}/seed-42/result-cam.json" for f in range(4)}
    if any(not p.is_file() or e05_identity.get("original_C1_report_sha256", {}).get(f) != sha256(p) for f,p in references.items()):
        raise ValueError("Original C1 reports are missing or differ from E05")
    raw_entries = {}
    for image_id in ids:
        stem = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:24]
        path, meta_path = e05 / "raw" / f"{stem}.npz", e05 / "raw" / f"{stem}.json"
        if not path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"E05 raw maps missing for {image_id}")
        meta = read_json(meta_path)
        if meta.get("image_id") != image_id or meta.get("sha256") != sha256(path):
            raise ValueError(f"E05 raw map integrity mismatch: {image_id}")
        raw_entries[image_id] = {"path": path, "sha256": meta["sha256"]}
    runtime = _runtime(config.device)
    print(f"E03: {len(ids)} development images; loading GeoRSCLIP only", flush=True)
    geo = GeoRSCLIP(config.georsclip_weights, config.georsclip_sha256, config.device)
    if geo.open_clip_version != cache_identity["open_clip_version"]:
        raise ValueError("OpenCLIP version differs from E05")
    identity = {"protocol": PROTOCOL, "runtime": runtime,
        "e05_identity_sha256": sha256(e05_identity_path), "e05_summary_sha256": sha256(e05_summary_path),
        "e05_raw_sha256": {i: raw_entries[i]["sha256"] for i in ids},
        "manifest_sha256": sha256(root / "dataset_manifest.json"), "split_sha256": sha256(root / "split_lock.json"),
        "original_C1_report_sha256": {f: sha256(p) for f,p in references.items()},
        "georsclip_sha256": geo.checkpoint_sha256, "open_clip_version": geo.open_clip_version,
        "code_sha256": {name: sha256(Path(__file__).parent/name) for name in
                        ("hard_negative_probe.py", "encoders.py", "semantic_probe.py", "priors.py", "metrics.py")}}
    folder = root / "diagnostics" / run_name
    folder.mkdir(parents=True, exist_ok=True)
    identity_path = folder / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("E03 identity changed; keep the old run and choose a new --run-name")
    _atomic_json(identity_path, identity)
    raw_folder = folder / "raw"
    raw_folder.mkdir(exist_ok=True)
    records, timing_rows = [], []
    for number, image_id in enumerate(ids, 1):
        row = rows[image_id]
        image_path = _checked_input_path(config.image_dir, image_id)
        label_path = _checked_input_path(config.label_dir, image_id)
        if sha256(image_path) != row["image_sha256"] or sha256(label_path) != row["label_sha256"] or index["entries"][image_id]["image_sha256"] != row["image_sha256"]:
            raise ValueError(f"Input changed after original audit: {image_id}")
        with np.load(raw_entries[image_id]["path"], allow_pickle=False) as source:
            full_native, full_sliding = source["native512"].copy(), source["sliding224"].copy()
        stem = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:24]
        raw_path, meta_path = raw_folder / f"{stem}.npz", raw_folder / f"{stem}.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            if meta.get("image_id") != image_id or meta.get("e05_raw_sha256") != raw_entries[image_id]["sha256"] or meta.get("sha256") != sha256(raw_path):
                raise ValueError(f"E03 raw map integrity mismatch: {image_id}")
            with np.load(raw_path, allow_pickle=False) as archive:
                positive_native = archive["native_positive_only"].copy()
                positive_sliding = archive["sliding_positive_only"].copy()
            print(f"[{number}/{len(ids)}] reuse {image_id}", flush=True)
        else:
            with rasterio.open(image_path) as image:
                rgb = torch.from_numpy(image.read().copy())[None]
            pair_native, duration_native, peak_native = _measure(lambda: geo.encode_semantic_pair(rgb), config.device)
            pair_sliding, duration_sliding, peak_sliding = _measure(lambda: geo.encode_semantic_pair(rgb, sliding224=True), config.device)
            fresh_native, positive_native_tensor = pair_native
            fresh_sliding, positive_sliding_tensor = pair_sliding
            if (not np.allclose(fresh_native[0,0].cpu().numpy(), full_native, rtol=1e-4, atol=1e-5) or
                not np.allclose(fresh_sliding[0,0].cpu().numpy(), full_sliding, rtol=1e-4, atol=1e-5)):
                raise ValueError(f"E03 full-text path differs from original E05 maps: {image_id}")
            positive_native = positive_native_tensor[0,0].cpu().numpy().copy()
            positive_sliding = positive_sliding_tensor[0,0].cpu().numpy().copy()
            temporary = raw_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, native_positive_only=positive_native,
                                sliding_positive_only=positive_sliding)
            temporary.replace(raw_path)
            meta = {"image_id": image_id, "e05_raw_sha256": raw_entries[image_id]["sha256"],
                    "sha256": sha256(raw_path), "full_text_matches_e05": True,
                    "seconds": {"native_pair": duration_native, "sliding_pair": duration_sliding},
                    "cuda_peak_allocated_bytes": {"native_pair": peak_native, "sliding_pair": peak_sliding}}
            _atomic_json(meta_path, meta)
            print(f"[{number}/{len(ids)}] extracted {image_id}", flush=True)
        if any(v.shape != (32,32) or not np.isfinite(v).all() for v in
               (full_native, full_sliding, positive_native, positive_sliding)):
            raise ValueError(f"E03 score map shape/value invalid: {image_id}")
        with rasterio.open(label_path) as label:
            mask = label.read(1).astype(np.float32)
        records.append({"id": image_id, "site": split["assignment"][image_id]["site"], "mask": mask,
                        "y32": area_target(mask), "native_full": full_native,
                        "native_positive_only": positive_native, "sliding_full": full_sliding,
                        "sliding_positive_only": positive_sliding})
        timing_rows.append(meta)
    result, folds, predictions = evaluate_e03(records, split)
    for fold in folds:
        e05_fold = read_json(e05 / f"fold-{fold['fold']}.json")
        old = read_json(references[str(fold["fold"])])
        if old["train_ids"] != fold["train_ids"] or old["validation_ids"] != fold["validation_ids"]:
            raise ValueError("Original C1 report split differs from E03")
        if e05_fold["train_ids"] != fold["train_ids"] or e05_fold["validation_ids"] != fold["validation_ids"]:
            raise ValueError("E05 fold split differs from E03")
        for current_mode, old_mode in (("native_full", "native512"), ("sliding_full", "sliding224")):
            current = {r["site"]: r for r in fold["metrics"][current_mode]["site_rows"]}
            previous = {r["site"]: r for r in e05_fold["metrics"][old_mode]["site_rows"]}
            _same_metrics(current, previous, context=f"E05/{current_mode}/fold-{fold['fold']}")
            if current_mode == "native_full":
                original = {r["site"]: r for r in old["cam"]["C1"]["site_rows"]}
                _same_metrics(current, original, context=f"original C1/fold-{fold['fold']}")
        _atomic_json(folder / f"fold-{fold['fold']}.json", fold)
    result.update({"status": "development_hard_negative_probe_complete", "schema": PROTOCOL["schema"],
                   "protocol": PROTOCOL, "development_images": len(ids), "runtime": runtime,
                   "e05_run_name": e05_run_name, "identity_sha256": sha256(identity_path),
                   "original_C1_reproduced_folds": list(range(4)), "E05_full_text_reproduced_folds": list(range(4)),
                   "timing": {"mean_seconds_per_image": {
                       key: float(np.mean([r["seconds"][key] for r in timing_rows])) for key in ("native_pair", "sliding_pair")}},
                   "limits": "E03 development diagnostic only. Native512 is the registered primary comparison; sliding224 is a conditional E05 follow-up. No model selection, main-cache overwrite or test use."})
    result["previews"] = _preview(config, records, rows, predictions, folder)
    _atomic_json(folder / "e03_summary.json", result)
    for mode in MODES[1:]:
        target = folder / "predictions" / mode
        target.mkdir(parents=True, exist_ok=True)
        for image_id, score in predictions[mode].items():
            stem = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:24]
            np.savez_compressed(target / f"{stem}.npz", probability=score)
    bundle = folder / "e03_review.zip"
    temporary = bundle.with_suffix(".tmp.zip")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for path in sorted(folder.glob("*.json")) + sorted(folder.glob("preview-*.png")):
            archive.write(path, path.name)
    temporary.replace(bundle)
    print(f"E03 complete: {folder / 'e03_summary.json'}\nReview bundle: {bundle}", flush=True)
    return result
