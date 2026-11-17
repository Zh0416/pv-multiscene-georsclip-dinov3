from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .cache import load_record, read_cache_index
from .config import Config
from .io import read_json, sha256, write_json
from .metrics import area_calibration, site_metrics
from .models import Decoder, FusionGate, fusion_loss, segmentation_loss
from .priors import area_target, bank_raw, build_bank, fit_prior, site_weights
from .splits import partition_ids


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor(record: dict, key: str, device: str) -> torch.Tensor:
    value = torch.as_tensor(record[key], device=device, dtype=torch.float32)
    if key in ("p_sem", "p_spa", "p_bank", "p_self", "M", "y32", "mask"):
        return value[None, None]
    if key == "rgb":
        return value[None] / 255
    if key == "dino":
        return value[None]
    return value.reshape(1)


def _gate_forward(model: FusionGate, r: dict, device: str) -> torch.Tensor:
    return model(_tensor(r, "dino", device), _tensor(r, "p_sem", device),
                 _tensor(r, "p_spa", device), _tensor(r, "s_global", device))[0]


def _train(model: nn.Module, train: list[dict], val: list[dict], loss_fn, *,
           seed: int, epochs: int, lr: float, weight_decay: float, device: str) -> tuple[nn.Module, dict]:
    _seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    best, best_epoch, stale = float("inf"), 0, 0
    state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(train))
        optimizer.zero_grad(set_to_none=True)
        train_losses = []
        for step, index in enumerate(order):
            with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
                loss = loss_fn(model, train[index])
            train_losses.append(float(loss.detach()))
            group_size = min(4, len(order) - 4 * (step // 4))
            scaler.scale(loss / group_size).backward()
            if (step + 1) % 4 == 0 or step + 1 == len(order):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            validation = float(np.mean([float(loss_fn(model, r)) for r in val]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": validation})
        if validation < best - 1e-6:
            best, best_epoch, stale = validation, epoch, 0
            state = copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()})
        else:
            stale += 1
        if stale >= 12:
            break
    if state is None:
        raise RuntimeError("Training produced no finite validation checkpoint")
    model.load_state_dict(state)
    return model, {"best_epoch": best_epoch, "best_validation_loss": best, "history": history}


def train_gate(train: list[dict], val: list[dict], mode: str, theta: float, *, seed: int,
               epochs: int, lr: float, weight_decay: float, device: str) -> tuple[FusionGate, dict]:
    def loss(model, r):
        return fusion_loss(_gate_forward(model, r, device), _tensor(r, "y32", device))
    return _train(FusionGate(mode, theta), train, val, loss, seed=seed, epochs=epochs,
                  lr=lr, weight_decay=weight_decay, device=device)


@torch.no_grad()
def add_gate_map(model: FusionGate, records: list[dict], key: str, device: str) -> None:
    model.eval()
    for r in records:
        m, details = model(_tensor(r, "dino", device), _tensor(r, "p_sem", device),
                           _tensor(r, "p_spa", device), _tensor(r, "s_global", device))
        r[key] = m[0, 0].float().cpu().numpy()
        if key == "C6":
            for name in ("a", "rho", "R", "route"):
                r[name] = details[name][0].float().cpu().numpy()


def _variant_prior(record: dict, variant: str) -> np.ndarray:
    if variant == "S2":
        return np.zeros((32, 32), dtype=np.float32)
    mapping = {"S3": "p_sem", "S4": "p_spa", "S6": "C6", "S7": "C6", "S11": "C7",
               "S13": "C5", "S14": "C5g"}
    if variant == "S5":
        return 0.5 * (record["p_sem"] + record["p_spa"])
    if variant not in mapping:
        raise ValueError(f"Unsupported decoder variant: {variant}")
    return record[mapping[variant]]


def train_decoder(train: list[dict], val: list[dict], variant: str, *, seed: int,
                  epochs: int, lr: float, weight_decay: float, device: str) -> tuple[Decoder, dict]:
    def loss(model, r):
        prior = torch.as_tensor(_variant_prior(r, variant), dtype=torch.float32, device=device)[None, None]
        logits = model(_tensor(r, "rgb", device), _tensor(r, "dino", device), prior, variant == "S7")
        return segmentation_loss(logits, _tensor(r, "mask", device))
    return _train(Decoder(), train, val, loss, seed=seed, epochs=epochs,
                  lr=lr, weight_decay=weight_decay, device=device)


@torch.no_grad()
def decoder_predictions(model: Decoder, records: list[dict], variant: str, device: str) -> dict[str, np.ndarray]:
    model.eval()
    results = {}
    for r in records:
        prior = torch.as_tensor(_variant_prior(r, variant), dtype=torch.float32, device=device)[None, None]
        logits = model(_tensor(r, "rgb", device), _tensor(r, "dino", device), prior, variant == "S7")
        results[r["id"]] = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
    return results


def _cam_predictions(records: list[dict], train: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    prevalence = float(np.sum([r["y32"].mean() * weight for r, weight in zip(train, site_weights(train, pixels=False))]))
    keys = {"C0a": lambda r: np.full((32, 32), prevalence, np.float32),
            "C1": lambda r: r["p_sem"], "C2": lambda r: r["p_bank"],
            "C2m": lambda r: r["p_spa"],
            "C3": lambda r: r["p_self"] if r["seed_count"] else r["p_bank"],
            "C4": lambda r: 0.5 * (r["p_sem"] + r["p_spa"]),
            "C5": lambda r: r["C5"], "C5g": lambda r: r["C5g"],
            "C6": lambda r: r["C6"], "C7": lambda r: r["C7"]}
    result = {name: {r["id"]: getter(r) for r in records} for name, getter in keys.items()}
    result["C0b"] = {}
    for positive in (False, True):
        group = sorted((r for r in records if bool(r["mask"].any()) == positive), key=lambda r: r["id"])
        shifts = [shift for shift in range(1, len(group))
                  if all(group[i]["site"] != group[(i + shift) % len(group)]["site"] for i in range(len(group)))]
        if not shifts:
            result.pop("C0b")
            break
        for i, record in enumerate(group):
            result["C0b"][record["id"]] = group[(i + shifts[0]) % len(group)]["p_sem"]
    return result


def _without_semantics(records: list[dict], semantic_mean: float, global_mean: float) -> list[dict]:
    return [{**r, "p_sem": np.full((32, 32), semantic_mean, dtype=np.float32),
             "p_spa": r["p_bank"], "s_global": global_mean} for r in records]


def _bank_drift(train: list[dict], val: list[dict], prior, purity: float, k: int, seed: int) -> dict:
    sites = sorted({r["site"] for r in train})
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(sites, size=max(4, round(len(sites) * 0.75)), replace=False))
    subbank, _ = build_bank([r for r in train if r["site"] in chosen], purity, k, seed)
    full = {r["id"]: r["p_bank"] for r in val}
    subset = {r["id"]: prior.bank.predict(bank_raw(r, subbank)) for r in val}
    full_metrics = area_calibration(val, full)
    subset_metrics = area_calibration(val, subset)
    delta_brier = full_metrics["area_brier"] - subset_metrics["area_brier"]
    delta_ece = full_metrics["area_ece"] - subset_metrics["area_ece"]
    return {"full_minus_subbank_brier": delta_brier, "full_minus_subbank_ece": delta_ece,
            "quality_pass": delta_brier <= 0.02 and delta_ece <= 0.05,
            "subset_sites": sorted(chosen), "full": full_metrics, "subset": subset_metrics}


def _save_predictions(folder: Path, records: list[dict], predictions: dict[str, np.ndarray]) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for r in records:
        path = folder / (hashlib.sha256(r["id"].encode("utf-8")).hexdigest()[:24] + ".npz")
        np.savez_compressed(path, probability=predictions[r["id"]].astype(np.float32))
        files[r["id"]] = {"file": path.name, "sha256": sha256(path)}
    return files


def _identity(manifest: dict, split: dict, index: dict, settings: dict) -> str:
    payload = {"image_hashes": [(r["id"], r["image_sha256"], r["label_sha256"]) for r in manifest["images"]],
               "assignment": split["assignment"], "split_status": split["status"],
               "cache_identity": index["identity"], "cache_entries": index["entries"],
               "settings": settings}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _save_prior_maps(folder: Path, records: list[dict]) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    fields = ("p_sem", "p_spa", "p_bank", "p_self", "C5", "C5g", "C6", "C7",
              "a", "rho", "R", "route", "s_global", "r_seed", "seed_count")
    saved = {}
    for record in records:
        path = folder / (hashlib.sha256(record["id"].encode("utf-8")).hexdigest()[:24] + ".npz")
        np.savez_compressed(path, **{key: np.asarray(record[key], dtype=np.float32) for key in fields})
        saved[record["id"]] = {"file": path.name, "sha256": sha256(path)}
    return saved


def _load_prior_maps(folder: Path, raw: list[dict], entries: dict) -> list[dict]:
    if {r["id"] for r in raw} != entries.keys():
        raise ValueError("Prior cache image IDs differ from the locked fold")
    result = []
    for record in raw:
        info = entries[record["id"]]
        path = folder / info["file"]
        if path.resolve().parent != folder.resolve() or sha256(path) != info["sha256"]:
            raise ValueError("Prior cache integrity mismatch")
        with np.load(path, allow_pickle=False) as archive:
            maps = {key: archive[key].copy() for key in archive.files}
        result.append({**record, **maps, "y32": area_target(record["mask"])})
    return result


def _route_summary(records: list[dict]) -> dict:
    result = {}
    for name, selected in (("positive", [r for r in records if r["mask"].any()]),
                           ("negative", [r for r in records if not r["mask"].any()])):
        if selected:
            result[name] = {key: float(np.mean([r[key].mean() for r in selected]))
                            for key in ("a", "rho", "R")}
            result[name]["route_mean"] = np.mean([r["route"].mean(axis=(1, 2)) for r in selected], axis=0).tolist()
            result[name]["max_route_sum_error"] = max(float(np.abs(r["route"].sum(0) - 1).max()) for r in selected)
    return result


def run_fold(config: Config, manifest: dict, split: dict, *, fold: int, seed: int,
             variant: str | None, epochs: int = 100, lr: float = 1e-4,
             weight_decay: float = 1e-4, purity: float = 0.5, k: int = 4) -> dict:
    if variant is not None and variant not in ("S2", "S3", "S4", "S5", "S6", "S7", "S11", "S13", "S14"):
        raise ValueError("Variant not implemented; refusing to relabel another model")
    if split["status"] != "locked_after_manual_qc":
        raise PermissionError("Manual QC and split lock are required before development training")
    if config.semantic_mode == "window_cls_positive_only":
        report_path = config.output_dir / "e08_cache_verification.json"
        if not report_path.is_file():
            raise FileNotFoundError("Run verify-e08-cache before E08 development training")
        report = read_json(report_path)
        if (report.get("status") != "e08_cache_verified" or
            report.get("cache_index_sha256") != sha256(config.output_dir / "cache" / "index.json") or
            report.get("manifest_sha256") != sha256(config.output_dir / "dataset_manifest.json") or
            report.get("split_sha256") != sha256(config.output_dir / "split_lock.json") or
            report.get("diagnostic_summary_sha256") != config.semantic_summary_sha256):
            raise ValueError("E08 cache verification is stale or differs from configured diagnostic")
    index = read_cache_index(config)
    train_ids = partition_ids(split, "train", fold)
    val_ids = partition_ids(split, "validation", fold)
    train_raw = [load_record(config, manifest, index, split, name) for name in train_ids]
    val_raw = [load_record(config, manifest, index, split, name) for name in val_ids]
    folder = config.output_dir / f"fold-{fold}" / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    settings = {"epochs_max": epochs, "lr": lr, "weight_decay": weight_decay,
                "purity": purity, "prototypes": k, "effective_batch": 4}
    identity = _identity(manifest, split, index, settings)
    ready_path = folder / "cam_ready.json"
    if ready_path.is_file():
        ready = read_json(ready_path)
        if ready["identity"] != identity:
            raise ValueError("Existing fold CAM uses different data, split, cache, or hyperparameters")
        train = _load_prior_maps(folder / "prior_train", train_raw, ready["train_maps"])
        validation = _load_prior_maps(folder / "prior_validation", val_raw, ready["validation_maps"])
    else:
        prior = None
        for inner_seed in (seed, 20261118, 20261119):
            try:
                prior, train = fit_prior(train_raw, purity=purity, k=k, seed=inner_seed)
                break
            except ValueError as error:
                if inner_seed == 20261119:
                    raise RuntimeError("G2 failed: all registered inner partitions failed") from error
        assert prior is not None
        validation = [prior.predict(r) for r in val_raw]
        bank_drift = _bank_drift(train, validation, prior, purity, k, seed)
        gates = {}
        for mode in ("C5", "C5g", "C6"):
            _seed(seed)
            gate, history = train_gate(train, validation, mode, prior.theta, seed=seed,
                                       epochs=epochs, lr=lr, weight_decay=weight_decay, device=config.device)
            torch.save(gate.state_dict(), folder / f"{mode}.pt")
            add_gate_map(gate, train, mode, config.device)
            add_gate_map(gate, validation, mode, config.device)
            gates[mode] = history
        weights = site_weights(train, pixels=False)
        semantic_mean = float(np.sum([r["p_sem"].mean() * w for r, w in zip(train, weights)]))
        global_mean = float(np.sum([r["s_global"] * w for r, w in zip(train, weights)]))
        c7_train = _without_semantics(train, semantic_mean, global_mean)
        c7_val = _without_semantics(validation, semantic_mean, global_mean)
        _seed(seed)
        c7_gate, c7_history = train_gate(c7_train, c7_val, "C6", prior.theta, seed=seed,
                                          epochs=epochs, lr=lr, weight_decay=weight_decay, device=config.device)
        torch.save(c7_gate.state_dict(), folder / "C7.pt")
        for source, target in ((c7_train, train), (c7_val, validation)):
            add_gate_map(c7_gate, source, "C7", config.device)
            for output, original in zip(source, target):
                original["C7"] = output["C7"]
        gates["C7"] = c7_history
        cam = {}
        for name, predictions in _cam_predictions(validation, train).items():
            cam[name] = {**site_metrics(validation, predictions, cam=True),
                         **area_calibration(validation, predictions),
                         "prediction_files": _save_predictions(folder / name, validation, predictions)}
        ready = {"identity": identity, "inner_seed": inner_seed, "prior": prior.diagnostics,
                 "bank_drift": bank_drift, "gates": gates, "cam": cam,
                 "route_summary": _route_summary(validation),
                 "train_maps": _save_prior_maps(folder / "prior_train", train),
                 "validation_maps": _save_prior_maps(folder / "prior_validation", validation)}
        write_json(ready_path, ready)
    segmentation = None
    if variant:
        _seed(seed)
        decoder, history = train_decoder(train, validation, variant, seed=seed,
                                         epochs=epochs, lr=lr, weight_decay=weight_decay, device=config.device)
        torch.save(decoder.state_dict(), folder / f"{variant}.pt")
        predictions = decoder_predictions(decoder, validation, variant, config.device)
        segmentation = {**site_metrics(validation, predictions, cam=False),
                        "prediction_files": _save_predictions(folder / variant, validation, predictions),
                        "training": history, "parameters": sum(p.numel() for p in decoder.parameters())}
    result = {"schema": "pvstudy-fold-v1", "fold": fold, "seed": seed, "variant": variant,
              "semantic_mode": config.semantic_mode,
              "train_ids": train_ids, "validation_ids": val_ids, "inner_seed": ready["inner_seed"],
              "training_config": settings, "cam_identity": identity,
              "prior": ready["prior"], "bank_drift": ready["bank_drift"],
              "gates": ready["gates"], "cam": ready["cam"],
              "route_summary": ready["route_summary"], "segmentation": segmentation,
              "status": "development_only", "augmentation": config.augmentation}
    write_json(folder / f"result-{variant or 'cam'}.json", result)
    return result
