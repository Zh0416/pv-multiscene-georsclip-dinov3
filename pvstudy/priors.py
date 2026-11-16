from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch.nn import functional as F


def area_target(mask: np.ndarray) -> np.ndarray:
    if mask.shape != (512, 512) or not np.isin(mask, (0, 1)).all():
        raise ValueError("Expected binary 512x512 mask")
    return mask.reshape(32, 16, 32, 16).mean(axis=(1, 3)).astype(np.float32)


def site_weights(records: list[dict], pixels: bool = True) -> np.ndarray:
    sites = {r["site"] for r in records}
    counts = {site: sum(r["site"] == site for r in records) for site in sites}
    per_image = np.array([1 / (len(sites) * counts[r["site"]]) for r in records], dtype=np.float64)
    if pixels:
        return np.repeat(per_image / 1024, 1024)
    return per_image


@dataclass(frozen=True)
class Calibrator:
    bias: float
    temperature: float
    success: bool
    hit_bound: bool
    objective: float

    def predict(self, score: np.ndarray | float) -> np.ndarray:
        return expit((np.asarray(score) - self.bias) / self.temperature).astype(np.float32)

    def as_dict(self) -> dict:
        return vars(self)


def fit_calibrator(score: np.ndarray, target: np.ndarray, weights: np.ndarray, *, global_score: bool = False) -> Calibrator:
    x, y, w = [np.asarray(v, dtype=np.float64).reshape(-1) for v in (score, target, weights)]
    if len(x) == 0 or x.shape != y.shape or x.shape != w.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError("Invalid calibration arrays or site weights")
    t_min, t_max = (0.02, 100.0) if global_score else (0.02, 1.0)
    low, high = float(x.min()), float(x.max())
    if low == high:
        low -= 1e-6
        high += 1e-6

    def objective(params):
        b, log_t = params
        t = np.exp(log_t)
        z = (x - b) / t
        p = expit(z)
        loss = np.sum(w * (np.logaddexp(0, z) - y * z))
        loss += 1e-3 * (b * b + (log_t - np.log(0.1)) ** 2)
        dz = w * (p - y)
        db = -dz.sum() / t + 2e-3 * b
        dlog_t = -(dz * z).sum() + 2e-3 * (log_t - np.log(0.1))
        return loss, np.array([db, dlog_t])

    result = minimize(objective, [float(np.median(x)), np.log(0.1)], jac=True,
                      bounds=[(low, high), (np.log(t_min), np.log(t_max))], method="L-BFGS-B")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Calibration optimization failed: {result.message}")
    b, log_t = result.x
    temp = float(np.exp(log_t))
    return Calibrator(float(b), temp, bool(result.success),
                      temp <= t_min * 1.001 or temp >= t_max * 0.999 or abs(b-low) < 1e-6 or abs(b-high) < 1e-6,
                      float(result.fun))


def _flat(records: list[dict], key: str, target: str = "y32") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.concatenate([r[key].reshape(-1) for r in records]),
            np.concatenate([r[target].reshape(-1) for r in records]), site_weights(records))


def fit_patch(records: list[dict], key: str) -> Calibrator:
    return fit_calibrator(*_flat(records, key))


def fit_global(records: list[dict]) -> Calibrator:
    return fit_calibrator(np.array([r["global_raw"] for r in records]),
                          np.array([r["mask"].any() for r in records], dtype=float),
                          site_weights(records, pixels=False), global_score=True)


def _normalized_features(record: dict) -> np.ndarray:
    return F.normalize(torch.from_numpy(record["dino"]).float(), dim=0).numpy().reshape(1024, 1024).T


def build_bank(records: list[dict], purity: float = 0.5, k: int = 4, seed: int = 42) -> tuple[np.ndarray, dict]:
    if k not in (1, 4, 8) or purity not in (0.5, 0.7):
        raise ValueError("Unregistered prototype configuration")
    rng = np.random.default_rng(seed)
    selected = []
    counts = {}
    for site in sorted({r["site"] for r in records}):
        candidates = []
        for record in records:
            if record["site"] == site:
                candidates.extend(_normalized_features(record)[(record["y32"] >= purity).ravel()])
        if candidates:
            candidates = np.asarray(candidates, dtype=np.float32)
            pick = rng.choice(len(candidates), size=min(64, len(candidates)), replace=False)
            selected.append(candidates[pick])
            counts[site] = len(pick)
    if not selected:
        raise ValueError("No positive pure patches in training sites")
    pool = np.concatenate(selected)
    if len(pool) < k:
        raise ValueError("Insufficient prototype patches")
    centers = pool[rng.choice(len(pool), k, replace=False)].copy()
    for _ in range(30):
        assignments = (pool @ centers.T).argmax(axis=1)
        new = centers.copy()
        for j in range(k):
            members = pool[assignments == j]
            if len(members):
                value = members.mean(0)
                new[j] = value / max(np.linalg.norm(value), 1e-12)
        if np.allclose(new, centers, atol=1e-5):
            break
        centers = new
    covered = sum(r["mask"].any() and (r["y32"] >= purity).any() for r in records)
    positives = sum(r["mask"].any() for r in records)
    return centers, {"patches_by_site": counts, "positive_image_coverage": covered / max(1, positives)}


def bank_raw(record: dict, centers: np.ndarray) -> np.ndarray:
    return (_normalized_features(record) @ centers.T).max(1).reshape(32, 32)


def select_seed_threshold(records: list[dict]) -> tuple[float, dict]:
    choices = []
    positive = [r for r in records if r["mask"].any()]
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        seeded = [r for r in records if (r["p_sem"] >= threshold).any()]
        coverage = sum((r["p_sem"] >= threshold).any() for r in positive) / max(1, len(positive))
        purity = float(np.mean([r["y32"][r["p_sem"] >= threshold].mean() for r in seeded])) if seeded else 0.0
        choices.append((threshold, coverage, purity))
    valid = [c for c in choices if c[1] >= 0.8]
    best = min(valid, key=lambda c: (-c[2], c[0])) if valid else min(choices, key=lambda c: (-c[1], -c[2], c[0]))
    return best[0], {"coverage": best[1], "purity": best[2], "quality_pass": bool(valid), "candidates": choices}


def self_raw(record: dict, threshold: float) -> tuple[np.ndarray, float, int]:
    scores = record["p_sem"].ravel()
    candidates = np.flatnonzero(scores >= threshold)
    if not len(candidates):
        return np.zeros((32, 32), dtype=np.float32), 0.0, 0
    selected = candidates[np.argsort(-scores[candidates], kind="stable")[:32]]
    values = scores[selected]
    weights = np.exp((values - values.max()) / 0.1)
    weights /= weights.sum()
    feature = _normalized_features(record)
    prototype = (feature[selected] * weights[:, None]).sum(0)
    prototype /= max(np.linalg.norm(prototype), 1e-12)
    reliability = np.clip((values.mean() - threshold) / (1 - threshold), 0, 1) * min(len(selected) / 4, 1)
    return (feature @ prototype).reshape(32, 32), float(reliability), len(selected)


def select_theta(records: list[dict]) -> tuple[float, dict]:
    candidates = []
    for theta in np.arange(0.2, 0.81, 0.1):
        candidates_pos, positives, neg_area = 0, 0, []
        for r in records:
            p = torch.from_numpy(r["p_sem"])[None, None]
            smooth = F.avg_pool2d(F.pad(p, (1, 1, 1, 1), mode="replicate"), 3, stride=1)[0, 0].numpy()
            region = 0.15 + 0.85 * expit(10 * (smooth - theta)) >= 0.5
            truth = r["y32"] >= 0.25
            candidates_pos += int(np.logical_and(region, truth).sum())
            positives += int(truth.sum())
            if not r["mask"].any():
                neg_area.append(float(region.mean()))
        candidates.append((round(float(theta), 1), candidates_pos / max(1, positives), float(np.mean(neg_area)) if neg_area else 0.0))
    valid = [c for c in candidates if c[1] >= 0.9]
    best = min(valid, key=lambda c: (c[2], -c[1], c[0])) if valid else min(candidates, key=lambda c: (-c[1], c[2], c[0]))
    return best[0], {"candidate_recall": best[1], "negative_area": best[2], "quality_pass": bool(valid), "candidates": candidates}


def _inner_groups(records: list[dict], seed: int) -> list[list[dict]]:
    sites = sorted({r["site"] for r in records})
    if len(sites) < 4:
        raise ValueError("At least four training sites required for OOF prior")
    rng = np.random.default_rng(seed)
    rng.shuffle(sites)
    return [[r for r in records if r["site"] in sites[i::4]] for i in range(4)]


@dataclass
class PriorModel:
    semantic: Calibrator
    global_score: Calibrator
    bank: Calibrator
    self_score: Calibrator
    spatial: Calibrator | None
    centers: np.ndarray
    seed_threshold: float
    theta: float
    diagnostics: dict

    def predict(self, record: dict) -> dict:
        output = dict(record)
        if "mask" in record:
            output["y32"] = area_target(record["mask"])
        output["p_sem"] = self.semantic.predict(record["semantic_raw"])
        output["s_global"] = float(self.global_score.predict(record["global_raw"]))
        output["bank_raw"] = bank_raw(record, self.centers)
        output["p_bank"] = self.bank.predict(output["bank_raw"])
        raw, reliability, count = self_raw(output, self.seed_threshold)
        output["p_self"] = self.self_score.predict(raw) if count else np.zeros((32, 32), dtype=np.float32)
        output["r_seed"] = reliability
        output["seed_count"] = count
        output["p_spa"] = (1 - 0.8 * reliability) * output["p_bank"] + 0.8 * reliability * output["p_self"]
        if self.spatial is not None:
            output["p_spa"] = self.spatial.predict(output["p_spa"])
        return output


def fit_prior(records: list[dict], *, purity: float = 0.5, k: int = 4, seed: int = 42) -> tuple[PriorModel, list[dict]]:
    prepared = [{**r, "y32": area_target(r["mask"])} for r in records]
    inner = _inner_groups(prepared, seed)
    oof = []
    for held in inner:
        train = [r for r in prepared if r["site"] not in {h["site"] for h in held}]
        if (not train or not any(r["mask"].any() for r in train) or
            not any(not r["mask"].any() for r in train)):
            raise ValueError("Inner fold needs both positive bank and negative global-calibration sites")
        sem, glob = fit_patch(train, "semantic_raw"), fit_global(train)
        centers, _ = build_bank(train, purity, k, seed)
        for r in held:
            oof.append({**r, "p_sem": sem.predict(r["semantic_raw"]),
                        "s_global": float(glob.predict(r["global_raw"])),
                        "bank_raw": bank_raw(r, centers)})
    seed_threshold, seed_diag = select_seed_threshold(oof)
    theta, theta_diag = select_theta(oof)
    bank_cal = fit_patch(oof, "bank_raw")
    seeded = []
    for r in oof:
        r["p_bank"] = bank_cal.predict(r["bank_raw"])
        r["self_raw"], r["r_seed"], r["seed_count"] = self_raw(r, seed_threshold)
        if r["seed_count"]:
            seeded.append(r)
    if not seeded:
        raise ValueError("No OOF semantic seeds for self calibration")
    self_cal = fit_patch(seeded, "self_raw")
    for r in oof:
        r["p_self"] = self_cal.predict(r["self_raw"]) if r["seed_count"] else np.zeros((32, 32), dtype=np.float32)
        r["p_spa"] = (1 - 0.8 * r["r_seed"]) * r["p_bank"] + 0.8 * r["r_seed"] * r["p_self"]
    raw_spa, target, weights = _flat(oof, "p_spa")
    uncalibrated_brier = float(np.sum(weights * (raw_spa - target) ** 2))
    spatial_candidate = fit_calibrator(raw_spa, target, weights)
    calibrated_brier = float(np.sum(weights * (spatial_candidate.predict(raw_spa) - target) ** 2))
    spatial = spatial_candidate if calibrated_brier < uncalibrated_brier - 0.005 else None
    if spatial is not None:
        for r in oof:
            r["p_spa"] = spatial.predict(r["p_spa"])
    centers, bank_diag = build_bank(prepared, purity, k, seed)
    semantic_cal, global_cal = fit_patch(prepared, "semantic_raw"), fit_global(prepared)
    model = PriorModel(semantic_cal, global_cal, bank_cal,
                       self_cal, spatial, centers, seed_threshold, theta,
                       {"seed": seed_diag, "theta": theta_diag, "bank": bank_diag,
                        "spatial_brier_before_after": [uncalibrated_brier, calibrated_brier],
                        "calibration": {"semantic": semantic_cal.as_dict(), "global": global_cal.as_dict(),
                                        "bank": bank_cal.as_dict(), "self": self_cal.as_dict(),
                                        "spatial": spatial.as_dict() if spatial else None}})
    return model, oof
