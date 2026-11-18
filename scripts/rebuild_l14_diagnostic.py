"""Rebuild a missing L/14 diagnostic from the locked Kaggle dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-source", type=Path,
                        default=Path("/kaggle/input/datasets/zh0416/pvstudy-e05-recovery"))
    parser.add_argument("--run-name", default="l14-semantic-v3")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config_path = repo / "config.kaggle.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("georsclip_model") != "ViT-L-14-336":
        raise ValueError("The diagnostic requires the unchanged L/14-336 configuration")
    for key in ("image_dir", "label_dir", "georsclip_weights", "dinov3_weights"):
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"Configured Kaggle input is missing: {config[key]}")
    lock_source = args.lock_source
    if not lock_source.is_file() and not (lock_source / "E05_RECOVERY_INFO.json").is_file() and not all(
        (lock_source / name).is_file() for name in ("dataset_manifest.json", "split_lock.json")
    ):
        raise FileNotFoundError(
            f"Locked split is missing at {lock_source}. Attach the pvstudy-e05-recovery "
            "Kaggle input dataset or pass --lock-source PATH.")
    base = [sys.executable, "-m", "pvstudy.cli", "--config", str(config_path)]
    for command in (("prepare-source",),
                    ("reuse-lock", "--source", str(lock_source)),
                    ("verify-encoders", "--id", "正样本/PV-01.tif"),
                    ("diagnose-l14", "--run-name", args.run_name)):
        subprocess.run(base + list(command), cwd=repo, check=True)
    diagnostic = Path(config["output_dir"]) / "diagnostics" / args.run_name
    summary = diagnostic / "l14_summary.json"
    count = len(list((diagnostic / "raw").glob("*.npz")))
    if count != 79:
        raise RuntimeError(f"Rebuilt diagnostic has {count}/79 raw maps")
    print(f"Review bundle: {diagnostic / 'l14_review.zip'}", flush=True)
    print(f"Summary SHA-256: {hashlib.sha256(summary.read_bytes()).hexdigest()}", flush=True)
    print("Stop here for review; the pinned E08 development config still points to the earlier run.", flush=True)


if __name__ == "__main__":
    main()
