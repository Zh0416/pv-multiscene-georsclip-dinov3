from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import rasterio
import torch

from .cache import _source_identity, build_cache
from .analysis import development_gates
from .config import Config
from .data import build_manifest
from .encoders import load_encoders
from .experiment import run_fold
from .io import read_json, sha256, write_json
from .splits import build_split, partition_ids
from .source import prepare_dinov3_source
from .qc import export_qc, validate_qc


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pvstudy")
    p.add_argument("--config", type=Path, default=Path("config.json"))
    for name in ("image-dir", "label-dir", "georsclip-weights", "dinov3-weights", "dinov3-source", "output-dir", "device"):
        p.add_argument("--" + name)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-source", help="Get the pinned official DINOv3 source checkout")
    audit = sub.add_parser("audit", help="Write data manifest and provisional site split")
    audit.add_argument("--manual-groups", type=Path)
    sub.add_parser("export-qc", help="Export an offline image/label QC page and unconfirmed draft")
    lock = sub.add_parser("lock-split", help="Lock an audited split after independent manual QC")
    lock.add_argument("--qc-file", type=Path, required=True)
    verify = sub.add_parser("verify-encoders", help="Run G1 deterministic shape and freeze checks")
    verify.add_argument("--id", required=True)
    cache = sub.add_parser("cache", help="Cache frozen development-set features")
    cache.add_argument("--include-test", action="store_true")
    fold = sub.add_parser("run-fold", help="Train OOF CAMs and one decoder variant")
    fold.add_argument("--fold", type=int, choices=range(4), required=True)
    fold.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    fold.add_argument("--variant", choices=("S2", "S3", "S4", "S5", "S6", "S7", "S11", "S13", "S14"))
    fold.add_argument("--epochs", type=int, default=100)
    fold.add_argument("--lr", type=float, choices=(1e-4, 3e-4), default=1e-4)
    fold.add_argument("--weight-decay", type=float, choices=(1e-4, 1e-2), default=1e-4)
    fold.add_argument("--purity", type=float, choices=(0.5, 0.7), default=0.5)
    fold.add_argument("--prototypes", type=int, choices=(1, 4, 8), default=4)
    sub.add_parser("summarize-dev", help="Evaluate registered G2-G5 development gates")
    probe = sub.add_parser("diagnose-semantic", help="Compare native512 and sliding224 GeoRSCLIP on locked development folds")
    probe.add_argument("--run-name", default="e05-native224-v1")
    probe.add_argument("--recompute-native", action="store_true", help="Rebuild only semantic scores when original feature arrays are lost; requires four original C1 reports")
    hard_neg = sub.add_parser("diagnose-hard-negatives", help="E03: compare full versus positive-only semantic scores on locked development folds")
    hard_neg.add_argument("--e05-run-name", default="e05-recovered-native-v1")
    hard_neg.add_argument("--run-name", default="e03-no-hard-negatives-v1")
    restore = sub.add_parser("restore-e05", help="Restore locked E05 reference JSONs from a recovery ZIP or mounted directory")
    restore.add_argument("--source", type=Path, required=True)
    return p


def _config(args) -> Config:
    config = Config.load(args.config)
    overrides = {}
    for name in ("image_dir", "label_dir", "georsclip_weights", "dinov3_weights", "dinov3_source", "output_dir", "device"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value if name == "device" else Path(value).resolve()
    return replace(config, **overrides)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    config = _config(args)
    output = config.output_dir
    if args.command == "prepare-source":
        if config.dinov3_source_commit is None:
            raise ValueError("dinov3_source_commit must be configured")
        print(prepare_dinov3_source(config.dinov3_source, config.dinov3_source_commit))
    elif args.command == "audit":
        manifest = build_manifest(config.image_dir, config.label_dir)
        manual = read_json(args.manual_groups) if args.manual_groups else None
        split = build_split(manifest, manual)
        write_json(output / "dataset_manifest.json", manifest)
        write_json(output / "split_lock.json", split)
        print(f"Provisional G0 audit: {len(manifest['images'])} pairs; test={split['counts']['test']}; manual QC pending")
    elif args.command == "export-qc":
        manifest = read_json(output / "dataset_manifest.json")
        split = read_json(output / "split_lock.json")
        page = export_qc(config.image_dir, config.label_dir, manifest, split, output / "qc")
        print(f"Open {page} in a browser; QC remains unconfirmed until reviewed")
    elif args.command == "lock-split":
        manifest = read_json(output / "dataset_manifest.json")
        split = read_json(output / "split_lock.json")
        qc = read_json(args.qc_file)
        if split["status"] != "candidate_requires_manual_qc":
            raise ValueError("Only a provisional split may be locked")
        validate_qc(manifest, split, qc)
        split["status"] = "locked_after_manual_qc"
        split["qc_sha256"] = sha256(args.qc_file)
        manifest["manual_qc_confirmed"] = True
        manifest["manual_qc_sha256"] = split["qc_sha256"]
        write_json(output / "dataset_manifest.json", manifest)
        write_json(output / "split_lock.json", split)
        print("Split locked; test remains sealed until explicit final evaluation")
    elif args.command == "verify-encoders":
        manifest = read_json(output / "dataset_manifest.json")
        rows = {r["id"]: r for r in manifest["images"]}
        if args.id not in rows or sha256(config.image_dir / args.id) != rows[args.id]["image_sha256"]:
            raise ValueError("Probe image missing or changed")
        with rasterio.open(config.image_dir / args.id) as image:
            rgb = torch.from_numpy(image.read().copy())[None]
        geo, dino = load_encoders(config)
        with torch.no_grad():
            a, b = geo.encode(rgb)
            c = dino.encode(rgb)
            a2, b2 = geo.encode(rgb)
            c2 = dino.encode(rgb)
        if a.shape != (1, 1, 32, 32) or b.shape != (1,) or c.shape != (1, 1024, 32, 32):
            raise RuntimeError("G1 shape mismatch")
        if not (torch.allclose(a, a2) and torch.allclose(b, b2) and torch.allclose(c, c2)):
            raise RuntimeError("G1 non-deterministic encoder output")
        info = {"status": "interface_pass_semantic_quality_unverified", "image_id": args.id,
                "georsclip_checkpoint_sha256": geo.checkpoint_sha256,
                "dinov3_checkpoint_sha256": dino.checkpoint_sha256,
                "dinov3_source": _source_identity(config.dinov3_source, config.dinov3_source_commit),
                "dinov3_layers": list(dino.layer_indices),
                "open_clip_version": geo.open_clip_version,
                "semantic_shape": list(a.shape), "global_shape": list(b.shape), "dino_shape": list(c.shape),
                "device": config.device}
        if config.device.startswith("cuda"):
            info["cuda_peak_bytes"] = torch.cuda.max_memory_allocated()
        write_json(output / "g1_encoder_probe.json", info)
        print(info)
    elif args.command == "cache":
        manifest = read_json(output / "dataset_manifest.json")
        split = read_json(output / "split_lock.json")
        ids = partition_ids(split, "dev")
        if args.include_test:
            ids += partition_ids(split, "test", unlock_test=True)
        result = build_cache(config, manifest, ids)
        print(f"Cached {len(result['entries'])} images")
    elif args.command == "run-fold":
        manifest = read_json(output / "dataset_manifest.json")
        split = read_json(output / "split_lock.json")
        result = run_fold(config, manifest, split, fold=args.fold, seed=args.seed,
                          variant=args.variant, epochs=args.epochs, lr=args.lr,
                          weight_decay=args.weight_decay, purity=args.purity, k=args.prototypes)
        print(f"Wrote fold {result['fold']} seed {result['seed']} {result['variant'] or 'CAM'}")
    elif args.command == "summarize-dev":
        result = development_gates(output)
        write_json(output / "development_gates.json", result)
        print(result["status"])
    elif args.command == "diagnose-semantic":
        from .semantic_probe import run_semantic_probe
        run_semantic_probe(config, run_name=args.run_name, recompute_native=args.recompute_native)
    elif args.command == "diagnose-hard-negatives":
        from .hard_negative_probe import run_hard_negative_probe
        run_hard_negative_probe(config, e05_run_name=args.e05_run_name, run_name=args.run_name)
    elif args.command == "restore-e05":
        from .recovery import restore_e05
        print(restore_e05(args.source, output))


if __name__ == "__main__":
    main()
