"""Run the current L/14 E08 development matrix on Kaggle."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


CORE = ("S2", "S14", "S6", "S7")
EXTENDED = CORE + ("S3", "S4", "S5", "S11", "S13")
SEEDS = (42, 43, 44)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.kaggle.e08.json"))
    parser.add_argument("--stage", choices=("core", "extended"), default="core")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else repo / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("georsclip_model") != "ViT-L-14-336" or config.get("semantic_mode") != "window_cls_positive_only":
        raise ValueError("This runner requires the current L/14 E08 configuration")
    output = Path(config["output_dir"])
    diagnostic = Path(config["semantic_diagnostic_dir"])
    summary = diagnostic / "l14_summary.json"
    if not summary.is_file() or hashlib.sha256(summary.read_bytes()).hexdigest() != config["semantic_summary_sha256"]:
        raise FileNotFoundError("Reviewed L/14 diagnostic summary missing or changed")
    if len(list((diagnostic / "raw").glob("*.npz"))) != 79:
        raise FileNotFoundError("The reviewed L/14 E08 diagnostic needs 79 raw maps")
    source = diagnostic.parents[1]
    if not all((source / name).is_file() for name in ("dataset_manifest.json", "split_lock.json")):
        raise FileNotFoundError("The locked manifest and split are missing from the L/14 source output")
    base = [sys.executable, "-m", "pvstudy.cli", "--config", str(config_path)]

    def run(*parts: str) -> None:
        subprocess.run(base + list(parts), cwd=repo, check=True)

    run("prepare-source")
    run("reuse-lock", "--source", str(source))
    run("verify-encoders", "--id", "正样本/PV-01.tif")
    run("cache")
    run("verify-e08-cache")
    for variant in (None,) + (CORE if args.stage == "core" else EXTENDED):
        for fold in range(4):
            for seed in SEEDS:
                filename = f"result-{variant or 'cam'}.json"
                result = output / f"fold-{fold}" / f"seed-{seed}" / filename
                if result.is_file():
                    print(f"Reuse {result}", flush=True)
                    continue
                command = ["run-fold", "--fold", str(fold), "--seed", str(seed)]
                if variant:
                    command += ["--variant", variant]
                run(*command)
        if variant is None:
            run("summarize-cam")
    run("summarize-l14", "--stage", args.stage)
    run("export-l14-review", "--stage", args.stage)
    print(f"Save the Kaggle output or download {output / f'l14_development_{args.stage}_review.zip'}", flush=True)


if __name__ == "__main__":
    main()
