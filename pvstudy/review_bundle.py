"""Export portable development evidence without unsealing test data."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .analysis import FOLDS, SEEDS, L14_CORE, L14_EXTENDED, cam_development_gates, l14_development_report
from .io import read_json, sha256, write_json


def export_l14_review(output: Path, stage: str = "core") -> Path:
    report = l14_development_report(output, stage)
    if report["status"] != "l14_development_complete":
        raise FileNotFoundError(f"L/14 {stage} results are incomplete: {report['missing_result_files']}")
    variants = L14_CORE if stage == "core" else L14_EXTENDED
    report_name = f"l14_development_{stage}.json"
    write_json(output / report_name, report)
    write_json(output / "cam_development_gates.json", cam_development_gates(output))
    names = ["dataset_manifest.json", "split_lock.json", "cache/index.json",
             "e08_cache_verification.json", "g1_encoder_probe.json",
             "cam_development_gates.json", report_name]
    for fold in FOLDS:
        for seed in SEEDS:
            folder = f"fold-{fold}/seed-{seed}"
            names.append(f"{folder}/result-cam.json")
            names.extend(f"{folder}/result-{variant}.json" for variant in variants)
    missing = [name for name in names if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"L/14 review evidence is missing: {missing}")
    source_identity = read_json(output / "cache/index.json")["identity"]["semantic_source"]
    source = Path(source_identity["directory"])
    diagnostic_files = {"diagnostic/identity.json": source / "identity.json",
                        "diagnostic/l14_summary.json": source / "l14_summary.json"}
    if (not all(path.is_file() for path in diagnostic_files.values()) or
        sha256(diagnostic_files["diagnostic/identity.json"]) != source_identity["identity_sha256"] or
        sha256(diagnostic_files["diagnostic/l14_summary.json"]) != source_identity["summary_sha256"]):
        raise ValueError("Reviewed L/14 diagnostic files are missing or changed")
    bundle = output / f"l14_development_{stage}_review.zip"
    temporary = bundle.with_suffix(".tmp.zip")
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                            capture_output=True, text=True)
    tracked_code = ("analysis.py", "cache.py", "config.py", "e08_source.py",
                    "encoders.py", "experiment.py", "metrics.py", "models.py",
                    "priors.py", "review_bundle.py")
    manifest = {"schema": "pvstudy-l14-development-review-v1", "stage": stage,
                "limits": "JSON summaries and identities only; no raw predictions, features, weights or test results",
                "code_commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "code_sha256": {name: sha256(Path(__file__).parent / name) for name in tracked_code},
                "files": {**{name: sha256(output / name) for name in names},
                          **{name: sha256(path) for name, path in diagnostic_files.items()}}}
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(output / name, name)
        for name, path in diagnostic_files.items():
            archive.write(path, name)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    temporary.replace(bundle)
    return bundle
