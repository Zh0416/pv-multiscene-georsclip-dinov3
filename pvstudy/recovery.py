"""Restore a small E05 reference bundle without manufacturing missing features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from .semantic_probe import _check_split
from .splits import partition_ids

RECOVERY_SCHEMA = "pvstudy-e05-recovery-v1"
RECOVERY_INFO = "E05_RECOVERY_INFO.json"
RECOVERY_FILES = ("dataset_manifest.json", "split_lock.json", "cache/index.json") + tuple(
    f"fold-{f}/seed-42/result-cam.json" for f in range(4))


def read_bundle(source: Path) -> tuple[dict, dict[str, bytes]]:
    if source.is_file():
        with ZipFile(source) as archive:
            names = [i.filename for i in archive.infolist()]
            for name in (RECOVERY_INFO,) + RECOVERY_FILES:
                if names.count(name) != 1 or archive.getinfo(name).file_size > 8 * 1024 * 1024:
                    raise ValueError(f"Invalid recovery bundle member: {name}")
            metadata = json.loads(archive.read(RECOVERY_INFO))
            payload = {name: archive.read(name) for name in RECOVERY_FILES}
    else:
        metadata = json.loads((source / RECOVERY_INFO).read_bytes())
        payload = {name: (source/name).read_bytes() for name in RECOVERY_FILES}
    if metadata.get("schema") != RECOVERY_SCHEMA or set(metadata.get("files", {})) != set(RECOVERY_FILES):
        raise ValueError("Unrecognized E05 recovery bundle")
    for name, data in payload.items():
        if hashlib.sha256(data).hexdigest() != metadata["files"][name]:
            raise ValueError(f"Recovery checksum mismatch: {name}")
    manifest, split, index = [json.loads(payload[n]) for n in RECOVERY_FILES[:3]]
    ids = _check_split(manifest, split)
    rows = {r["id"]: r for r in manifest["images"]}
    if not set(ids) <= set(index["entries"]):
        raise ValueError("Recovery cache index lacks development IDs")
    for image_id in ids:
        if index["entries"][image_id]["image_sha256"] != rows[image_id]["image_sha256"]:
            raise ValueError("Recovery manifest/cache image identities differ")
    for fold in range(4):
        result = json.loads(payload[f"fold-{fold}/seed-42/result-cam.json"])
        if (result["fold"], result["seed"], result["variant"]) != (fold, 42, None):
            raise ValueError("Recovery CAM identity differs")
        if result["train_ids"] != partition_ids(split, "train", fold) or result["validation_ids"] != partition_ids(split, "validation", fold):
            raise ValueError("Recovery CAM uses another split")
        if not result["cam"]["C1"]["site_rows"]:
            raise ValueError("Recovery CAM has no original C1 metrics")
    return metadata, payload


def restore_e05(source: Path, output: Path) -> dict:
    if source.is_file() or (source / RECOVERY_INFO).is_file():
        candidates = [source]
    else:
        # Kaggle can mount the uploaded ZIP itself or extract it automatically.
        candidates = sorted({p.parent for p in source.rglob(RECOVERY_INFO)})
        if not candidates:
            candidates = sorted(source.rglob("pvstudy-e05-recovery.zip"))
    if not candidates:
        raise FileNotFoundError("Attach pvstudy-e05-recovery.zip as a Kaggle input dataset, then rerun restore-e05")
    bundles = [read_bundle(p) for p in candidates]
    if any(meta["files"] != bundles[0][0]["files"] for meta, _ in bundles[1:]):
        raise ValueError("Multiple different recovery bundles found; specify the exact --source path")
    metadata, payload = bundles[0]
    # Validate every destination before writing any file. Never overwrite a different run.
    for name, data in payload.items():
        target = (output/name).resolve()
        if output.resolve() not in target.parents:
            raise ValueError("Recovery destination escapes output directory")
        if target.exists() and (not target.is_file() or target.read_bytes() != data):
            raise ValueError(f"Existing output differs; preserved without overwrite: {target}")
    for name, data in payload.items():
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(".restore.tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
    return {"status": "e05_references_restored", "source": str(candidates[0]),
            "output_dir": str(output), "files": len(payload),
            "feature_arrays_restored": False,
            "next_command": "diagnose-semantic --recompute-native --run-name e05-recovered-native-v1"}


def reuse_locked_split(source: Path, output: Path) -> dict:
    """Copy only the audited lock into a new model-specific output directory."""
    names = ("dataset_manifest.json", "split_lock.json")
    if source.is_file() or (source / RECOVERY_INFO).is_file():
        _, bundle = read_bundle(source)
        payload = {name: bundle[name] for name in names}
    else:
        payload = {name: (source / name).read_bytes() for name in names}
    manifest, split = (json.loads(payload[name]) for name in names)
    _check_split(manifest, split)
    for name, data in payload.items():
        target = output / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f"Existing lock differs; preserved: {target}")
    for name, data in payload.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(".lock.tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
    return {"status": "locked_split_reused", "source": str(source), "output_dir": str(output),
            "manifest_sha256": hashlib.sha256(payload[names[0]]).hexdigest(),
            "split_sha256": hashlib.sha256(payload[names[1]]).hexdigest()}
