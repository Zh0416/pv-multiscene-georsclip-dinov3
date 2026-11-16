from __future__ import annotations

from collections import defaultdict

import numpy as np


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    labels, values = y.astype(bool).ravel(), score.ravel()
    if not labels.any():
        return float("nan")
    order = np.argsort(-values, kind="stable")
    labels, values = labels[order], values[order]
    last = np.r_[np.flatnonzero(np.diff(values)), len(values) - 1]
    tp = np.cumsum(labels)[last]
    precision = tp / (last + 1)
    recall_step = np.diff(np.r_[0, tp]) / labels.sum()
    return float(np.sum(precision * recall_step))


def dice_iou(y: np.ndarray, score: np.ndarray, threshold: float) -> tuple[float, float]:
    truth, prediction = y.astype(bool), score >= threshold
    tp = int(np.logical_and(truth, prediction).sum())
    fp = int(np.logical_and(~truth, prediction).sum())
    fn = int(np.logical_and(truth, ~prediction).sum())
    return ((2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.0,
            (tp / (tp + fp + fn)) if tp + fp + fn else 1.0)


def site_metrics(records: list[dict], predictions: dict[str, np.ndarray], *, cam: bool) -> dict:
    by_site = defaultdict(list)
    for record in records:
        by_site[record["site"]].append(record)
    rows = []
    for site, items in sorted(by_site.items()):
        positives, negatives = [], []
        site_truth, site_scores = [], []
        for r in items:
            y = r["y32"] if cam else r["mask"]
            p = predictions[r["id"]]
            if p.shape != y.shape or not np.isfinite(p).all():
                raise ValueError("Prediction shape/value mismatch")
            if r["mask"].any():
                truth = y >= 0.25 if cam else y
                d, iou = dice_iou(truth, p, 0.25 if cam else 0.5)
                positives.append({"dice": d, "iou": iou})
                site_truth.append(truth.ravel())
                site_scores.append(p.ravel())
            else:
                negatives.append(float((p >= (0.25 if cam else 0.5)).mean()))
        rows.append({"site": site, "positive_images": len(positives), "negative_images": len(negatives),
                     "dice": float(np.mean([v["dice"] for v in positives])) if positives else None,
                     "iou": float(np.mean([v["iou"] for v in positives])) if positives else None,
                     "ap": average_precision(np.concatenate(site_truth), np.concatenate(site_scores)) if positives else None,
                     "negative_fp": float(np.mean(negatives)) if negatives else None})
    def macro(key: str) -> float | None:
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(values)) if values else None
    return {"site_rows": rows, "site_macro_dice": macro("dice"), "site_macro_iou": macro("iou"),
            "site_macro_ap": macro("ap"), "negative_site_fp": macro("negative_fp")}


def area_calibration(records: list[dict], predictions: dict[str, np.ndarray]) -> dict:
    y = np.concatenate([r["y32"].ravel() for r in records])
    p = np.concatenate([predictions[r["id"]].ravel() for r in records])
    if not (0 <= p).all() or not (p <= 1).all():
        raise ValueError("Area predictions must lie in [0,1]")
    bins = np.minimum((p * 10).astype(int), 9)
    calibration = []
    ece = 0.0
    for b in range(10):
        selected = bins == b
        if selected.any():
            mean_p, mean_y = float(p[selected].mean()), float(y[selected].mean())
            ece += selected.mean() * abs(mean_p - mean_y)
            calibration.append({"bin": b, "count": int(selected.sum()), "predicted_area": mean_p, "true_area": mean_y})
    return {"area_brier": float(np.mean((p-y)**2)), "area_mae": float(np.mean(abs(p-y))),
            "area_ece": float(ece), "bins": calibration}


def paired_site_bootstrap(a: dict[str, float], b: dict[str, float], seed: int = 20261120, draws: int = 10000) -> dict:
    if a.keys() != b.keys() or not a:
        raise ValueError("Paired site metric requires identical nonempty site IDs")
    sites = sorted(a)
    differences = np.array([a[s] - b[s] for s in sites], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sites), size=(draws, len(sites)))
    samples = differences[indices].mean(1)
    return {"sites": len(sites), "mean_delta": float(differences.mean()),
            "ci95_percentile": [float(v) for v in np.quantile(samples, [0.025, 0.975])],
            "positive_site_fraction": float((differences > 0).mean()),
            "worst_site": sites[int(differences.argmin())], "worst_delta": float(differences.min()),
            "site_differences": dict(zip(sites, differences.tolist()))}
