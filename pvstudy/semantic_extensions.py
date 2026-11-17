"""Isolated E04 and exploratory E08 diagnostics on locked development sites."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw

from .cache import PREPROCESS_VERSION, read_cache_index
from .config import Config
from .encoders import GeoRSCLIP, NEGATIVE_CLASSES, POSITIVE_PROMPTS
from .hard_negative_probe import _same_metrics
from .io import read_json, sha256
from .metrics import area_calibration, paired_site_bootstrap, site_metrics
from .priors import area_target, fit_patch, site_weights
from .semantic_probe import _atomic_json, _check_split, _checked_input_path, _measure, _runtime
from .splits import partition_ids

PROTOCOLS = {
    "e04": {
        "schema": "pvstudy-e04-v1", "last_modified_blocks": 1,
        "modes": ["constant", "native_full", "value_only_full", "vv_full",
                  "native_positive_only", "value_only_positive_only", "vv_positive_only"],
        "primary": "full-text modes compare with native_full; positive-only modes are exploratory E03 follow-up",
        "value_only": "last block patch V projection -> out_proj -> residual -> ln_2/MLP -> residual -> ln_post/proj",
        "vv": "last block patch-only per-head SDPA(V,V,V) -> out_proj -> residual -> ln_2/MLP -> residual -> ln_post/proj",
        "global": "native unmodified 512 CLS path; candidate patch outputs never replace it",
        "replacement": "full-text candidate minus native AP >=0.01; candidate minus native negative FP <=0.005; finite outputs and unchanged global CLS",
    },
    "e08": {
        "schema": "pvstudy-e08-v1", "modes": ["constant", "native_positive_only",
                                                     "sliding_positive_only", "window_cls_positive_only"],
        "window": 128, "stride": 64, "starts": [0, 64, 128, 192, 256, 320, 384],
        "window_batch": 7, "resize": "bilinear to 224x224, align_corners=False, before CLIP normalization",
        "stitch": "uniform average of raw CLS cosine over each covered 8x8 area cell",
        "text": "E03 positive mean only", "status": "new exploratory method, not part of original E04/E05 registration",
    },
}
COMMON = {
    "calibration": "independent site-weighted sigmoid fit on each outer training fold only",
    "target": "512 binary mask area-averaged to 32x32; positive CAM truth >=0.25",
    "threshold": 0.25, "preview": "smallest/median/largest positive fraction and first negative ID",
    "scope": "79 locked development images only; no decoder or main-cache replacement",
}


def evaluate_extension(records: list[dict], split: dict, modes: list[str]) -> tuple[dict, list[dict], dict]:
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(partition_ids(split, "dev")):
        raise ValueError("Extension evaluation requires exactly the locked development records")
    by_id = {r["id"]: r for r in records}
    if any(r["site"] != split["assignment"][r["id"]]["site"] for r in records):
        raise ValueError("Extension record site differs from locked split")
    folds, rows = [], {mode: [] for mode in modes}
    predictions = {mode: {} for mode in modes}
    for fold in range(4):
        train = [by_id[i] for i in partition_ids(split, "train", fold)]
        validation = [by_id[i] for i in partition_ids(split, "validation", fold)]
        prevalence = float(sum(r["y32"].mean() * w for r, w in zip(train, site_weights(train, pixels=False))))
        calibrators, metrics = {}, {}
        for mode in modes:
            if mode == "constant":
                scores = {r["id"]: np.full((32, 32), prevalence, np.float32) for r in validation}
            else:
                calibrator = fit_patch(train, mode)
                calibrators[mode] = calibrator.as_dict()
                scores = {r["id"]: calibrator.predict(r[mode]) for r in validation}
            metrics[mode] = {**site_metrics(validation, scores, cam=True),
                             **area_calibration(validation, scores)}
            rows[mode].extend(metrics[mode]["site_rows"])
            predictions[mode].update(scores)
        folds.append({"fold": fold, "train_ids": [r["id"] for r in train],
                      "validation_ids": [r["id"] for r in validation],
                      "calibrators": calibrators, "metrics": metrics})
    aggregate = {}
    for mode, site_rows in rows.items():
        if len(site_rows) != len({r["site"] for r in site_rows}):
            raise ValueError("Extension site appears in more than one validation fold")
        aggregate[mode] = {key: float(np.mean([r[key] for r in site_rows if r[key] is not None]))
                           for key in ("ap", "dice", "iou", "negative_fp")}
        if not all(np.isfinite(v) for v in aggregate[mode].values()):
            raise ValueError("Extension requires positive and negative development sites")
    e01, comparisons = {}, {}
    for mode in modes[1:]:
        delta = aggregate[mode]["ap"] - aggregate["constant"]["ap"]
        positive_folds = sum(f["metrics"][mode]["site_macro_ap"] > f["metrics"]["constant"]["site_macro_ap"] for f in folds)
        e01[mode] = {"ap_delta_vs_constant": delta, "positive_folds": positive_folds,
                     "quality_pass": delta >= 0.05 and positive_folds >= 3}
    for mode in modes[1:]:
        reference = ("native_full" if mode.endswith("_full") else "native_positive_only")
        if mode == reference:
            continue
        left = {r["site"]: r["ap"] for r in rows[mode] if r["ap"] is not None}
        right = {r["site"]: r["ap"] for r in rows[reference] if r["ap"] is not None}
        comparisons[mode] = {"reference": reference,
            "ap": paired_site_bootstrap(left, right),
            "negative_fp_delta": aggregate[mode]["negative_fp"] - aggregate[reference]["negative_fp"],
            "fold_ap_deltas": [f["metrics"][mode]["site_macro_ap"] - f["metrics"][reference]["site_macro_ap"] for f in folds]}
        if mode.endswith("_full"):
            comparisons[mode]["E04_replacement_pass"] = (
                comparisons[mode]["ap"]["mean_delta"] >= 0.01 and
                comparisons[mode]["negative_fp_delta"] <= 0.005)
    if "window_cls_positive_only" in modes:
        mode, reference = "window_cls_positive_only", "sliding_positive_only"
        left = {r["site"]: r["ap"] for r in rows[mode] if r["ap"] is not None}
        right = {r["site"]: r["ap"] for r in rows[reference] if r["ap"] is not None}
        comparisons["window_cls_vs_e05_positive_only"] = {
            "reference": reference, "ap": paired_site_bootstrap(left, right),
            "negative_fp_delta": aggregate[mode]["negative_fp"] - aggregate[reference]["negative_fp"],
            "fold_ap_deltas": [f["metrics"][mode]["site_macro_ap"] - f["metrics"][reference]["site_macro_ap"] for f in folds]}
    return {"aggregate": aggregate, "E01": e01, "comparisons": comparisons}, folds, predictions


def _previews(config: Config, records: list[dict], rows: dict, predictions: dict,
              modes: list[str], folder: Path) -> list[dict]:
    positive = sorted((r for r in records if r["mask"].any()),
                      key=lambda r: (rows[r["id"]]["target_fraction"], r["id"]))
    negative = sorted((r for r in records if not r["mask"].any()), key=lambda r: r["id"])
    selected = (("smallest_positive", positive[0]), ("median_positive", positive[len(positive)//2]),
                ("largest_positive", positive[-1]), ("first_negative", negative[0]))
    exported = []
    for reason, record in selected:
        with rasterio.open(_checked_input_path(config.image_dir, record["id"])) as image:
            rgb = Image.fromarray(image.read().transpose(1, 2, 0))
        panels = [(rgb, "RGB"), (Image.fromarray((record["mask"]*255).astype(np.uint8)).convert("RGB"), "Ground truth")]
        for mode in modes[1:]:
            p = np.clip(predictions[mode][record["id"]], 0, 1)
            color = np.stack((p, np.zeros_like(p), 1-p), axis=-1)
            panels.append((Image.fromarray(np.round(color*255).astype(np.uint8)).resize((512,512), Image.Resampling.NEAREST), mode))
        width, height = 1536, 540 * ((len(panels) + 2) // 3)
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (panel, label) in enumerate(panels):
            x, y = (i % 3)*512, (i // 3)*540
            draw.text((x+8,y+6), label, fill="black")
            canvas.paste(panel, (x,y+26))
        path = folder / f"preview-{reason}.png"
        canvas.save(path)
        exported.append({"selection_rule": reason, "image_id": record["id"], "file": path.name,
                         "color_scale": "fixed blue=0 red=1, no per-image normalization"})
    return exported


def run_extension(config: Config, *, method: str, e03_run_name: str = "e03-no-hard-negatives-v1",
                  run_name: str | None = None) -> dict:
    if method not in PROTOCOLS:
        raise ValueError("Unknown semantic extension")
    run_name = run_name or ("e04-last-block-v1" if method == "e04" else "e08-window-cls-v1")
    for name in (run_name, e03_run_name):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name):
            raise ValueError("Diagnostic run names must be short and alphanumeric")
    if run_name == e03_run_name:
        raise ValueError("Extension output must differ from E03")
    root = config.output_dir
    manifest_path, split_path = root / "dataset_manifest.json", root / "split_lock.json"
    manifest, split = read_json(manifest_path), read_json(split_path)
    ids = _check_split(manifest, split)
    index = read_cache_index(config)
    cache_identity = index["identity"]
    expected = {"preprocess": PREPROCESS_VERSION, "augmentation": "none",
                "georsclip_sha256": config.georsclip_sha256,
                "prompts_positive": list(POSITIVE_PROMPTS), "prompts_negative": list(NEGATIVE_CLASSES),
                "georsclip_dense": "native patch tokens, bicubic 7x7 positional interpolation to 16x16"}
    if config.augmentation != "none" or any(cache_identity.get(k) != v for k, v in expected.items()):
        raise ValueError("Original cache identity differs from registered baseline")
    e03 = root / "diagnostics" / e03_run_name
    e03_identity_path, e03_summary_path = e03 / "identity.json", e03 / "e03_summary.json"
    if not e03_identity_path.is_file() or not e03_summary_path.is_file():
        raise FileNotFoundError("Complete E03 diagnostic directory required, including raw maps")
    e03_identity, e03_summary = read_json(e03_identity_path), read_json(e03_summary_path)
    if (e03_summary.get("status") != "development_hard_negative_probe_complete" or
        e03_summary.get("development_images") != len(ids) or
        e03_summary.get("identity_sha256") != sha256(e03_identity_path) or
        e03_identity.get("manifest_sha256") != sha256(manifest_path) or
        e03_identity.get("split_sha256") != sha256(split_path) or
        e03_identity.get("georsclip_sha256") != config.georsclip_sha256):
        raise ValueError("E03 reference does not match locked development experiment")
    e05_name = e03_summary["e05_run_name"]
    e05 = root / "diagnostics" / e05_name
    e05_identity_path, e05_summary_path = e05 / "identity.json", e05 / "e05_summary.json"
    if (not e05_identity_path.is_file() or
        not e05_summary_path.is_file() or
        e03_identity.get("e05_identity_sha256") != sha256(e05_identity_path) or
        e03_identity.get("e05_summary_sha256") != sha256(e05_summary_path) or
        read_json(e05_identity_path).get("cache_identity") != cache_identity):
        raise ValueError("E05 cache/reference identity differs from E03")
    references = {str(f): root / f"fold-{f}/seed-42/result-cam.json" for f in range(4)}
    if any(not path.is_file() or e03_identity.get("original_C1_report_sha256", {}).get(f) != sha256(path)
           for f, path in references.items()):
        raise ValueError("Original C1 reports differ from E03")
    raw_entries = {}
    for image_id in ids:
        stem = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:24]
        e03_raw, e03_meta = e03 / "raw" / f"{stem}.npz", e03 / "raw" / f"{stem}.json"
        e05_raw, e05_meta = e05 / "raw" / f"{stem}.npz", e05 / "raw" / f"{stem}.json"
        if not all(p.is_file() for p in (e03_raw, e03_meta, e05_raw, e05_meta)):
            raise FileNotFoundError(f"E03/E05 raw reference missing: {image_id}")
        if (read_json(e03_meta).get("sha256") != sha256(e03_raw) or
            read_json(e05_meta).get("sha256") != sha256(e05_raw) or
            e03_identity.get("e05_raw_sha256", {}).get(image_id) != sha256(e05_raw)):
            raise ValueError(f"E03/E05 raw reference integrity mismatch: {image_id}")
        raw_entries[image_id] = (e03_raw, e05_raw)
    runtime = _runtime(config.device)
    geo = GeoRSCLIP(config.georsclip_weights, config.georsclip_sha256, config.device)
    if geo.open_clip_version != cache_identity.get("open_clip_version"):
        raise ValueError("OpenCLIP version differs from original cache")
    protocol = {**PROTOCOLS[method], **COMMON}
    identity = {"protocol": protocol, "runtime": runtime, "method": method,
        "e03_identity_sha256": sha256(e03_identity_path), "e03_summary_sha256": sha256(e03_summary_path),
        "e03_raw_sha256": {i: sha256(raw_entries[i][0]) for i in ids},
        "e05_raw_sha256": {i: sha256(raw_entries[i][1]) for i in ids},
        "manifest_sha256": sha256(manifest_path), "split_sha256": sha256(split_path),
        "original_C1_report_sha256": {f: sha256(p) for f, p in references.items()},
        "georsclip_sha256": geo.checkpoint_sha256, "open_clip_version": geo.open_clip_version,
        "code_sha256": {name: sha256(Path(__file__).parent/name) for name in
                        ("semantic_extensions.py", "encoders.py", "priors.py", "metrics.py")}}
    folder = root / "diagnostics" / run_name
    folder.mkdir(parents=True, exist_ok=True)
    identity_path = folder / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Diagnostic identity changed; preserve old run and use a new --run-name")
    _atomic_json(identity_path, identity)
    raw_folder = folder / "raw"
    raw_folder.mkdir(exist_ok=True)
    rows = {r["id"]: r for r in manifest["images"]}
    records, timings = [], []
    candidate_modes = (["value_only_full", "vv_full", "value_only_positive_only", "vv_positive_only"]
                       if method == "e04" else ["window_cls_positive_only"])
    for number, image_id in enumerate(ids, 1):
        row = rows[image_id]
        image_path = _checked_input_path(config.image_dir, image_id)
        label_path = _checked_input_path(config.label_dir, image_id)
        if (sha256(image_path) != row["image_sha256"] or sha256(label_path) != row["label_sha256"] or
            index["entries"][image_id]["image_sha256"] != row["image_sha256"]):
            raise ValueError(f"Input changed after original audit: {image_id}")
        with np.load(raw_entries[image_id][0], allow_pickle=False) as source:
            native_positive = source["native_positive_only"].copy()
            sliding_positive = source["sliding_positive_only"].copy()
        with np.load(raw_entries[image_id][1], allow_pickle=False) as source:
            native_full = source["native512"].copy()
        stem = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:24]
        raw_path, meta_path = raw_folder / f"{stem}.npz", raw_folder / f"{stem}.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            if (meta.get("image_id") != image_id or
                meta.get("e03_raw_sha256") != sha256(raw_entries[image_id][0]) or
                meta.get("sha256") != sha256(raw_path)):
                raise ValueError(f"Extension raw-map integrity mismatch: {image_id}")
            with np.load(raw_path, allow_pickle=False) as source:
                candidate = {mode: source[mode].copy() for mode in candidate_modes}
            print(f"[{number}/{len(ids)}] reuse {image_id}", flush=True)
        else:
            with rasterio.open(image_path) as image:
                rgb = torch.from_numpy(image.read().copy())[None]
            candidate, seconds, peak = {}, {}, {}
            if method == "e04":
                for name in ("value_only", "vv"):
                    result, seconds[name], peak[name] = _measure(
                        lambda name=name: geo.encode_last_block_pair(rgb, mode=name), config.device)
                    full, positive = result[name]
                    candidate[f"{name}_full"] = full[0, 0].cpu().numpy().copy()
                    candidate[f"{name}_positive_only"] = positive[0, 0].cpu().numpy().copy()
            else:
                result, seconds["window_cls"], peak["window_cls"] = _measure(
                    lambda: geo.encode_window_cls(rgb), config.device)
                candidate["window_cls_positive_only"] = result[0, 0].cpu().numpy().copy()
            native_result, seconds["native_cls_check"], peak["native_cls_check"] = _measure(
                lambda: geo.encode(rgb), config.device)
            if (not np.allclose(native_result[0][0, 0].cpu().numpy(), native_full, rtol=1e-4, atol=1e-5) or
                not np.isfinite(float(native_result[1][0]))):
                raise ValueError(f"Native path changed relative to E05: {image_id}")
            with np.load(raw_entries[image_id][1], allow_pickle=False) as source:
                if not np.isclose(float(native_result[1][0]), float(source["global_raw"]), rtol=1e-4, atol=1e-5):
                    raise ValueError(f"Native global CLS changed relative to E05: {image_id}")
            temporary = raw_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **candidate)
            temporary.replace(raw_path)
            meta = {"image_id": image_id, "e03_raw_sha256": sha256(raw_entries[image_id][0]),
                    "sha256": sha256(raw_path), "seconds": seconds,
                    "cuda_peak_allocated_bytes": peak, "native_cls_unchanged": True}
            _atomic_json(meta_path, meta)
            print(f"[{number}/{len(ids)}] extracted {image_id}", flush=True)
        if any(v.shape != (32, 32) or not np.isfinite(v).all() for v in
               (native_full, native_positive, sliding_positive, *candidate.values())):
            raise ValueError(f"Extension map shape/value invalid: {image_id}")
        with rasterio.open(label_path) as label:
            mask = label.read(1).astype(np.float32)
        record = {"id": image_id, "site": split["assignment"][image_id]["site"], "mask": mask,
                  "y32": area_target(mask), "native_full": native_full,
                  "native_positive_only": native_positive,
                  "sliding_positive_only": sliding_positive, **candidate}
        records.append(record)
        timings.append(meta)
    modes = protocol["modes"]
    result, folds, predictions = evaluate_extension(records, split, modes)
    for fold in folds:
        original = read_json(references[str(fold["fold"])])
        old_e03 = read_json(e03 / f"fold-{fold['fold']}.json")
        if (fold["train_ids"] != original["train_ids"] or
            fold["validation_ids"] != original["validation_ids"] or
            fold["train_ids"] != old_e03["train_ids"] or
            fold["validation_ids"] != old_e03["validation_ids"]):
            raise ValueError("Extension fold split differs from original C1/E03")
        for mode in modes[1:]:
            if mode not in old_e03["metrics"]:
                continue
            current = {r["site"]: r for r in fold["metrics"][mode]["site_rows"]}
            previous = {r["site"]: r for r in old_e03["metrics"][mode]["site_rows"]}
            _same_metrics(current, previous, context=f"E03/{mode}/fold-{fold['fold']}")
        _atomic_json(folder / f"fold-{fold['fold']}.json", fold)
    result.update({"schema": protocol["schema"], "status": "development_extension_complete",
                   "method": method, "protocol": protocol, "development_images": len(ids),
                   "identity_sha256": sha256(identity_path), "runtime": runtime,
                   "timing": {key: float(np.mean([t["seconds"][key] for t in timings]))
                              for key in timings[0]["seconds"]},
                   "peak_allocated_bytes": {key: max((t["cuda_peak_allocated_bytes"][key] or 0)
                                                     for t in timings) for key in timings[0]["seconds"]},
                   "reference_reproduced_folds": list(range(4)),
                   "limits": "Development diagnostic only; no model selection or test evaluation."})
    result["E03_E05_reference_aggregate"] = {
        mode: e03_summary["aggregate"][mode] for mode in
        ("constant", "native_full", "native_positive_only", "sliding_full", "sliding_positive_only")}
    result["previews"] = _previews(config, records, rows, predictions, modes, folder)
    summary_path = folder / f"{method}_summary.json"
    _atomic_json(summary_path, result)
    for mode in modes[1:]:
        target = folder / "predictions" / mode
        target.mkdir(parents=True, exist_ok=True)
        for image_id, score in predictions[mode].items():
            stem = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:24]
            np.savez_compressed(target / f"{stem}.npz", probability=score)
    bundle = folder / f"{method}_review.zip"
    temporary = bundle.with_suffix(".tmp.zip")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as zip_file:
        for path in (summary_path, identity_path, *(folder / f"fold-{i}.json" for i in range(4)),
                     *(folder / item["file"] for item in result["previews"])):
            zip_file.write(path, arcname=path.name)
    temporary.replace(bundle)
    print(f"{method.upper()} complete: {summary_path}\nReview bundle: {bundle}", flush=True)
    return result
