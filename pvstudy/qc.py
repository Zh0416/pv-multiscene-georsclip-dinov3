"""Image-only, pre-prediction manual quality review; never auto-approves QC."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from .io import sha256, write_json
from .data import footprint_distance_m


def review_identity(manifest: dict, split: dict) -> str:
    payload = {"images": sorted(manifest["images"], key=lambda row: row["id"]),
               "assignment": split["assignment"]}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_qc(manifest: dict, split: dict, qc: dict) -> None:
    ids = {row["id"] for row in manifest["images"]}
    reviewed = qc.get("reviewed_ids", [])
    if (not isinstance(reviewed, list) or not all(isinstance(v, str) for v in reviewed)
            or len(reviewed) != len(ids) or set(reviewed) != ids):
        raise ValueError("QC must list every audited image ID exactly once")
    for name in ("band_order_confirmed", "label_alignment_confirmed", "source_scene_checked"):
        if qc.get(name) is not True:
            raise ValueError(f"QC confirmation is missing: {name}")
    if qc.get("pending_issues"):
        raise ValueError("Resolve the pending image QC issues before locking")
    if (manifest["exact_duplicate_images"] or manifest["matching_thumbnail_fingerprints"]):
        if qc.get("duplicate_candidates_resolved") is not True:
            raise ValueError("Resolve duplicate candidates before locking")
    if "review_identity" in qc or qc.get("schema") == "pvstudy-manual-qc-v1":
        if qc.get("review_identity") != review_identity(manifest, split):
            raise ValueError("QC belongs to a different manifest or split; review the current version")


def _preview(array: np.ndarray, *, lossless: bool = False) -> str:
    buffer = io.BytesIO()
    image = Image.fromarray(array)
    if lossless:
        image.save(buffer, format="PNG")
        mime = "png"
    else:
        image.save(buffer, format="JPEG", quality=95, subsampling=0)
        mime = "jpeg"
    return f"data:image/{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _checked_path(root: Path, name: str, digest: str) -> Path:
    path = (root / name).resolve()
    if root.resolve() not in path.parents or sha256(path) != digest:
        raise ValueError(f"QC image/label path or SHA-256 mismatch: {name}")
    return path


def export_qc(image_dir: Path, label_dir: Path, manifest: dict, split: dict,
              output: Path) -> Path:
    if split.get("status") != "candidate_requires_manual_qc":
        raise ValueError("QC export requires the provisional pre-prediction split")
    ids = {row["id"] for row in manifest["images"]}
    if len(ids) != len(manifest["images"]) or set(split["assignment"]) != ids:
        raise ValueError("QC manifest and split IDs differ")
    output.mkdir(parents=True, exist_ok=True)
    identity = review_identity(manifest, split)
    template = {"schema": "pvstudy-manual-qc-v1", "review_identity": identity,
                "reviewed_ids": [], "band_order_confirmed": False,
                "label_alignment_confirmed": False, "source_scene_checked": False,
                "duplicate_candidates_resolved": False, "pending_issues": [], "notes": ""}
    entries = []
    for row in sorted(manifest["images"], key=lambda r: (not r["positive"], r["target_fraction"], r["id"])):
        name = row["id"]
        image_path = _checked_path(image_dir, name, row["image_sha256"])
        label_path = _checked_path(label_dir, name, row["label_sha256"])
        with rasterio.open(image_path) as src, rasterio.open(label_path) as dst:
            rgb, mask = src.read().transpose(1, 2, 0), dst.read(1)
        if rgb.shape != (512, 512, 3) or mask.shape != (512, 512) or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"Invalid QC image/label shape or values: {name}")
        overlay = rgb.copy()
        selected = mask.astype(bool)
        overlay[selected] = np.rint(0.6 * rgb[selected] + 0.4 * np.array([255, 30, 100])).astype(np.uint8)
        entries.append({"id": name, "positive": row["positive"], "fraction": row["target_fraction"],
                        **split["assignment"][name], "bounds": row["bounds"],
                        "rgb": _preview(rgb), "mask": _preview(mask * 255, lossless=True),
                        "overlay": _preview(overlay)})
    ten = split.get("ten_km_site_ids", {})
    cross_groups = []
    for group in sorted(set(ten.values())):
        names = sorted(name for name in ten if ten[name] == group)
        if len({split["assignment"][name]["partition"] for name in names}) > 1:
            cross_groups.append(names)
    cross_pairs = [(footprint_distance_m(left, right), left["id"], right["id"])
                   for i, left in enumerate(manifest["images"])
                   for right in manifest["images"][i + 1:]
                   if split["assignment"][left["id"]]["partition"] != split["assignment"][right["id"]]["partition"]]
    closest_pair = min(cross_pairs) if cross_pairs else None
    data = {"template": template, "entries": entries, "cross_groups": cross_groups,
            "closest_pair": closest_pair,
            "duplicate_candidates": manifest["exact_duplicate_images"] + manifest["matching_thumbnail_fingerprints"]}
    # Escape '<' so a filename cannot terminate the application/json element.
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    page = Path(__file__).with_name("qc.html").read_text(encoding="utf-8")
    page = page.replace("__QC_DATA__", payload)
    target = output / "qc_review.html"
    target.write_text(page, encoding="utf-8")
    draft = output / "QC.template.json"
    if not draft.exists():
        write_json(draft, template)
    return target
