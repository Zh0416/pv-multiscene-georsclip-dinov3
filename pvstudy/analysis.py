from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import read_json

SEEDS = (42, 43, 44)
FOLDS = range(4)


def load_grid(output: Path, variants: tuple[str, ...] = ("S2", "S6", "S7")) -> dict:
    grid = {}
    missing = []
    for variant in variants:
        for fold in FOLDS:
            for seed in SEEDS:
                path = output / f"fold-{fold}" / f"seed-{seed}" / f"result-{variant}.json"
                if not path.is_file():
                    missing.append(str(path))
                else:
                    result = read_json(path)
                    if (result["fold"], result["seed"], result["variant"]) != (fold, seed, variant):
                        raise ValueError(f"Result identity mismatch: {path}")
                    grid[(variant, fold, seed)] = result
    return {"results": grid, "missing": missing}


def _site_values(grid: dict, variant: str, metric: str, *, cam: bool) -> dict[str, float]:
    values = {}
    for fold in FOLDS:
        for seed in SEEDS:
            result = grid[(variant, fold, seed)]
            source = result["cam"][metric] if cam else result["segmentation"]
            if source is None:
                raise ValueError(f"Missing segmentation for {variant} fold {fold} seed {seed}")
            for row in source["site_rows"]:
                key = (row["site"], seed)
                if key in values:
                    raise ValueError("Site appeared in more than one development fold")
                values[key] = row
    return values


def _macro(values: dict, key: str) -> float:
    grouped = {}
    for (site, _seed), row in values.items():
        if row[key] is not None:
            grouped.setdefault(site, []).append(row[key])
    if any(len(samples) != 3 for samples in grouped.values()):
        raise ValueError("Each site needs all three seed measurements")
    return float(np.mean([np.mean(grouped[site]) for site in grouped])) if grouped else float("nan")


def _comparison(grid: dict, candidate: str, control: str, key: str, *, cam: bool) -> dict:
    left = _site_values(grid, "S2" if cam else candidate, candidate if cam else "", cam=cam)
    right = _site_values(grid, "S2" if cam else control, control if cam else "", cam=cam)
    if left.keys() != right.keys():
        raise ValueError("Comparison uses different site/seed pairs")
    fold_deltas = []
    for fold in FOLDS:
        seed_deltas = []
        for seed in SEEDS:
            a = grid[("S2" if cam else candidate, fold, seed)]
            b = grid[("S2" if cam else control, fold, seed)]
            a_rows = a["cam"][candidate]["site_rows"] if cam else a["segmentation"]["site_rows"]
            b_rows = b["cam"][control]["site_rows"] if cam else b["segmentation"]["site_rows"]
            amap = {r["site"]: r[key] for r in a_rows if r[key] is not None}
            bmap = {r["site"]: r[key] for r in b_rows if r[key] is not None}
            if amap.keys() != bmap.keys():
                raise ValueError("Fold sites differ")
            seed_deltas.append(float(np.mean([amap[s] - bmap[s] for s in amap])))
        fold_deltas.append(float(np.mean(seed_deltas)))
    delta = _macro(left, key) - _macro(right, key)
    fp_delta = _macro(left, "negative_fp") - _macro(right, "negative_fp")
    return {"delta": delta, "negative_fp_delta": fp_delta, "fold_deltas_three_seed_mean": fold_deltas,
            "positive_folds": sum(v > 0 for v in fold_deltas)}


def _check_identities(grid: dict, output: Path) -> str:
    identities = {result["cam_identity"] for result in grid.values()}
    if len(identities) != 1:
        raise ValueError("Development results have mixed cache or training identities")
    modes = {result.get("semantic_mode", "native_full") for result in grid.values()}
    if len(modes) != 1:
        raise ValueError("Development results have mixed semantic modes")
    mode = modes.pop()
    manifest_path, split_path, index_path = (output / name for name in
                                             ("dataset_manifest.json", "split_lock.json", "cache/index.json"))
    present = all(path.is_file() for path in (manifest_path, split_path, index_path))
    if mode == "window_cls_positive_only" and not present:
        raise FileNotFoundError("E08 gate summary requires the locked manifest, split and cache index")
    if present:
        from .experiment import _identity

        manifest, split, index = (read_json(path) for path in (manifest_path, split_path, index_path))
        if index["identity"].get("semantic_mode", "native_full") != mode:
            raise ValueError("Development result semantic mode differs from feature cache")
        if any(result["cam_identity"] != _identity(manifest, split, index, result["training_config"])
               for result in grid.values()):
            raise ValueError("Development result identity differs from current locked cache")
    return mode


def _cam_gates(grid: dict) -> dict:
    cam_names = ("C1", "C2m", "C4", "C5", "C5g")
    cam_ap = {name: _macro(_site_values(grid, "S2", name, cam=True), "ap") for name in cam_names}
    best_simple = max(cam_names, key=lambda name: (cam_ap[name], -cam_names.index(name)))
    g2_sem = _comparison(grid, "C1", "C0a", "ap", cam=True)
    g2_bank = _comparison(grid, "C2", "C0a", "ap", cam=True)
    g3 = _comparison(grid, "C6", best_simple, "ap", cam=True)
    cover = [grid[("S2", f, s)]["prior"]["bank"]["positive_image_coverage"] for f in FOLDS for s in SEEDS]
    drift = [grid[("S2", f, s)]["bank_drift"]["quality_pass"] for f in FOLDS for s in SEEDS]
    return {"best_simple_cam": best_simple,
            "G2": {"semantic": g2_sem, "bank": g2_bank,
                   "bank_coverage_min": min(cover), "bank_drift_all_pass": all(drift),
                   "quality_pass": all(c["delta"] >= 0.05 and c["positive_folds"] >= 3 for c in (g2_sem, g2_bank)) and min(cover) >= 0.95 and all(drift)},
            "G3": {**g3, "quality_pass": g3["delta"] >= 0.02 and g3["positive_folds"] >= 3 and g3["negative_fp_delta"] <= 0.005}}


def cam_development_gates(output: Path) -> dict:
    grid, missing = {}, []
    for fold in FOLDS:
        for seed in SEEDS:
            path = output / f"fold-{fold}" / f"seed-{seed}" / "result-cam.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            result = read_json(path)
            if (result["fold"], result["seed"], result["variant"]) != (fold, seed, None):
                raise ValueError(f"CAM result identity mismatch: {path}")
            grid[("S2", fold, seed)] = result
    if missing:
        return {"status": "incomplete", "missing_result_files": missing}
    mode = _check_identities(grid, output)
    return {"status": "development_cam_grid_complete", "semantic_mode": mode,
            **_cam_gates(grid),
            "limits": "CAM-only G2/G3 development gates; no decoder or test result inferred"}


def development_gates(output: Path) -> dict:
    loaded = load_grid(output)
    if loaded["missing"]:
        return {"status": "incomplete", "missing_result_files": loaded["missing"]}
    grid = loaded["results"]
    mode = _check_identities(grid, output)
    g4 = _comparison(grid, "S6", "S2", "dice", cam=False)
    g5 = _comparison(grid, "S7", "S6", "dice", cam=False)
    return {"status": "development_grid_complete", "semantic_mode": mode,
            **_cam_gates(grid),
            "G4": {**g4, "quality_pass": g4["delta"] >= 0.02 and g4["positive_folds"] >= 3 and g4["negative_fp_delta"] <= 0.005},
            "G5": {**g5, "quality_pass": g5["delta"] >= 0.01 and g5["positive_folds"] >= 3 and g5["negative_fp_delta"] <= 0.005},
            "limits": "G2-G5 require real encoder quality review and full baseline grid before G6; no test result inferred"}
