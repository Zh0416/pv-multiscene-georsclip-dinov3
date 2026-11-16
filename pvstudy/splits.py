from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .data import site_groups
from .io import sha256


def _group_rows(rows: list[dict], site_ids: dict[str, str]) -> list[dict]:
    members = defaultdict(list)
    for row in rows:
        members[site_ids[row["id"]]].append(row)
    return [{"site": site, "ids": [r["id"] for r in group], "n": len(group),
             "positive": sum(r["positive"] for r in group),
             "negative": sum(not r["positive"] for r in group),
             "small": sum(r["positive"] and r["target_fraction"] < 0.01 for r in group)}
            for site, group in sorted(members.items())]


def _test_score(groups: list[dict], chosen: set[str]) -> float:
    selected = [g for g in groups if g["site"] in chosen]
    total = {key: sum(g[key] for g in groups) for key in ("n", "positive", "negative", "small")}
    actual = {key: sum(g[key] for g in selected) for key in total}
    score = sum(((actual[k] - 0.2 * total[k]) / max(1, 0.2 * total[k])) ** 2 for k in total)
    pos_sites = sum(g["positive"] > 0 for g in selected)
    neg_sites = sum(g["negative"] > 0 for g in selected)
    if sum(g["positive"] > 0 for g in groups) >= 14:
        score += max(0, 10 - pos_sites) ** 2
    if sum(g["negative"] > 0 for g in groups) >= 7:
        score += max(0, 3 - neg_sites) ** 2
    return score


def _pick_test(groups: list[dict], seed: int) -> set[str]:
    rng = random.Random(seed)
    target = round(sum(g["n"] for g in groups) * 0.2)
    best, best_score = set(), float("inf")
    for _ in range(4000):
        shuffled = groups[:]
        rng.shuffle(shuffled)
        chosen = set()
        count = 0
        for group in shuffled:
            if count >= target and rng.random() < 0.85:
                continue
            chosen.add(group["site"])
            count += group["n"]
            if count >= target + 3:
                break
        score = _test_score(groups, chosen)
        if score < best_score:
            best, best_score = chosen, score
    return best


def _folds(groups: list[dict], seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    shuffled = groups[:]
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda g: (g["n"], g["negative"] > 0), reverse=True)
    targets = {k: sum(g[k] for g in groups) / 4 for k in ("n", "positive", "negative", "small")}
    bins = [{k: 0 for k in targets} for _ in range(4)]
    assignment = {}
    negative_anchors = [g for g in shuffled if g["negative"] > 0][:4]
    if len(negative_anchors) < 4:
        raise ValueError("Fewer than four negative development sites")
    anchor_folds = list(range(4))
    rng.shuffle(anchor_folds)
    for group, fold in zip(negative_anchors, anchor_folds):
        assignment[group["site"]] = fold
        for key in targets:
            bins[fold][key] += group[key]
    for group in shuffled:
        if group["site"] in assignment:
            continue
        order = list(range(4))
        rng.shuffle(order)
        def cost(i: int) -> float:
            return sum(
                ((bins[i][k] + group[k] - targets[k]) ** 2 - (bins[i][k] - targets[k]) ** 2)
                / max(1, targets[k]) ** 2 for k in targets
            )
        fold = min(order, key=cost)
        assignment[group["site"]] = fold
        for key in targets:
            bins[fold][key] += group[key]
    return assignment


def build_split(manifest: dict, manual_groups: list[list[str]] | None = None) -> dict:
    rows = manifest["images"]
    sites = site_groups(rows, 5, manual_groups)
    groups = _group_rows(rows, sites)
    selected = _pick_test(groups, 20261116)
    dev = [g for g in groups if g["site"] not in selected]
    folds = None
    used_seed = None
    for seed in (20261117, 20261118, 20261119):
        candidate = _folds(dev, seed)
        if all(any(g["negative"] for g in dev if candidate[g["site"]] == f) for f in range(4)):
            folds, used_seed = candidate, seed
            break
    if folds is None:
        raise ValueError("No preset fold seed placed a negative image in every validation fold")
    mapping = {row["id"]: {"site": sites[row["id"]], "partition": "test" if sites[row["id"]] in selected else "dev",
                            "fold": None if sites[row["id"]] in selected else folds[sites[row["id"]]]}
               for row in rows}
    assert all(len({mapping[r["id"]]["partition"] for r in rows if sites[r["id"]] == site}) == 1 for site in set(sites.values()))
    test = [r for r in rows if mapping[r["id"]]["partition"] == "test"]
    return {"schema": "pvstudy-split-v1", "status": "candidate_requires_manual_qc",
            "distance_km": 5, "test_seed": 20261116, "fold_seed": used_seed,
            "split_script_sha256": sha256(Path(__file__)), "assignment": mapping,
            "counts": {"total": len(rows), "test": len(test), "test_positive": sum(r["positive"] for r in test),
                       "test_negative": sum(not r["positive"] for r in test),
                       "site_count": len(groups), "test_sites": len(selected),
                       "folds": {str(f): sum(v["fold"] == f for v in mapping.values()) for f in range(4)}},
            "ten_km_site_ids": site_groups(rows, 10, manual_groups)}


def partition_ids(split: dict, partition: str, fold: int | None = None, unlock_test: bool = False) -> list[str]:
    if partition == "test" and (not unlock_test or split["status"] != "locked_after_manual_qc"):
        raise PermissionError("Test labels and predictions require an explicit locked split and unlock flag")
    if partition not in ("train", "validation", "dev", "test"):
        raise ValueError(partition)
    if partition in ("train", "validation") and fold not in range(4):
        raise ValueError("A development fold 0..3 is required")
    result = []
    for name, info in split["assignment"].items():
        if partition == "dev" and info["partition"] == "dev":
            result.append(name)
        elif partition == "test" and info["partition"] == "test":
            result.append(name)
        elif partition == "train" and info["partition"] == "dev" and info["fold"] != fold:
            result.append(name)
        elif partition == "validation" and info["fold"] == fold:
            result.append(name)
    return sorted(result)
