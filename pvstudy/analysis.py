from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import read_json, sha256
from .metrics import paired_site_bootstrap

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


L14_CORE = ("S2", "S14", "S6", "S7")
L14_EXTENDED = L14_CORE + ("S3", "S4", "S5", "S11", "S13")


def _paired_sites(grid: dict, candidate: str, control: str, key: str, *, cam: bool = False) -> dict:
    left = _site_values(grid, "S2" if cam else candidate, candidate if cam else "", cam=cam)
    right = _site_values(grid, "S2" if cam else control, control if cam else "", cam=cam)
    if left.keys() != right.keys():
        raise ValueError("Paired comparison uses different site/seed pairs")
    def site_means(rows: dict) -> dict[str, float]:
        grouped = {}
        for (site, _seed), row in rows.items():
            if row[key] is not None:
                grouped.setdefault(site, []).append(row[key])
        if any(len(values) != len(SEEDS) for values in grouped.values()):
            raise ValueError("Each paired site needs all three seeds")
        return {site: float(np.mean(values)) for site, values in grouped.items()}
    return paired_site_bootstrap(site_means(left), site_means(right))


def l14_development_report(output: Path, stage: str = "core") -> dict:
    if stage not in ("core", "extended"):
        raise ValueError("L/14 development stage must be core or extended")
    variants = L14_CORE if stage == "core" else L14_EXTENDED
    loaded = load_grid(output, variants=variants)
    cam_paths = [output / f"fold-{fold}" / f"seed-{seed}" / "result-cam.json"
                 for fold in FOLDS for seed in SEEDS]
    missing = loaded["missing"] + [str(path) for path in cam_paths if not path.is_file()]
    if missing:
        return {"status": "incomplete", "stage": stage, "missing_result_files": missing}
    grid = loaded["results"]
    cam_grid = {("S2", fold, seed): read_json(output / f"fold-{fold}" / f"seed-{seed}" / "result-cam.json")
                for fold in FOLDS for seed in SEEDS}
    if _check_identities({**grid, **{("CAM", f, s): value for (_, f, s), value in cam_grid.items()}}, output) != "window_cls_positive_only":
        raise ValueError("L/14 development report requires the verified E08 semantic mode")
    index = read_json(output / "cache" / "index.json")
    if index["identity"].get("georsclip_model") != "ViT-L-14-336":
        raise ValueError("L/14 development report requires GeoRSCLIP ViT-L-14-336")
    verification = read_json(output / "e08_cache_verification.json")
    if (verification.get("status") != "e08_cache_verified" or
        verification.get("cache_index_sha256") != sha256(output / "cache" / "index.json") or
        verification.get("manifest_sha256") != sha256(output / "dataset_manifest.json") or
        verification.get("split_sha256") != sha256(output / "split_lock.json")):
        raise ValueError("L/14 development report requires current E08 cache verification")
    cam = _cam_gates(cam_grid)
    cam["G2"]["semantic"]["paired_sites"] = _paired_sites(cam_grid, "C1", "C0a", "ap", cam=True)
    cam["G2"]["bank"]["paired_sites"] = _paired_sites(cam_grid, "C2", "C0a", "ap", cam=True)
    cam["G3"]["paired_sites"] = _paired_sites(cam_grid, "C6", cam["best_simple_cam"], "ap", cam=True)
    c6_vs_c7 = _comparison(cam_grid, "C6", "C7", "ap", cam=True)
    cam["C6_vs_C7"] = {**c6_vs_c7,
                       "quality_pass": c6_vs_c7["delta"] >= 0.01 and c6_vs_c7["positive_folds"] >= 3 and
                                       c6_vs_c7["negative_fp_delta"] <= 0.005,
                       "paired_sites": _paired_sites(cam_grid, "C6", "C7", "ap", cam=True)}
    segmentation = {}
    for variant in variants:
        rows = _site_values(grid, variant, "", cam=False)
        segmentation[variant] = {key: _macro(rows, key) for key in ("dice", "iou", "ap", "negative_fp")}
    pairs = [("S14", "S2"), ("S6", "S14"), ("S6", "S2"), ("S7", "S6")]
    if stage == "extended":
        pairs += [("S6", "S11"), ("S6", "S13"), ("S6", "S3"), ("S6", "S4"),
                  ("S6", "S5"), ("S14", "S13")]
    comparisons = {}
    for candidate, control in pairs:
        name = f"{candidate}_vs_{control}"
        comparison = _comparison(grid, candidate, control, "dice", cam=False)
        comparison["paired_sites"] = _paired_sites(grid, candidate, control, "dice")
        threshold = {"S6_vs_S14": 0.01, "S6_vs_S2": 0.02, "S7_vs_S6": 0.01,
                     "S6_vs_S11": 0.01, "S6_vs_S13": 0.01}.get(name)
        comparison["quality_pass"] = (comparison["delta"] >= threshold and
                                      comparison["positive_folds"] >= 3 and
                                      comparison["negative_fp_delta"] <= 0.005) if threshold is not None else None
        comparisons[name] = comparison
    return {"status": "l14_development_complete", "stage": stage,
            "semantic_mode": "window_cls_positive_only", "georsclip_model": "ViT-L-14-336",
            "variants": list(variants), "CAM": cam, "segmentation": segmentation,
            "comparisons": comparisons,
            "limits": "Development-only comparisons. Passing the registered gates and separate controls is required for full fusion claims; no final test or G6 inference."}
