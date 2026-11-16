from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Geod
from shapely.geometry import box
from shapely.ops import nearest_points

from .io import sha256

GEOD = Geod(ellps="WGS84")


def _files(root: Path) -> dict[str, Path]:
    paths = list(root.rglob("*.tif")) + list(root.rglob("*.tiff"))
    found = {str(p.relative_to(root)).replace("\\", "/"): p for p in paths}
    if len(found) != len(paths):
        raise ValueError("Duplicate relative image or label paths")
    return found


def build_manifest(image_dir: Path, label_dir: Path) -> dict:
    images, labels = _files(image_dir), _files(label_dir)
    if not images or images.keys() != labels.keys():
        raise ValueError(f"Unpaired TIFF paths: image-only={sorted(images.keys()-labels.keys())}, label-only={sorted(labels.keys()-images.keys())}")
    rows = []
    fingerprints = defaultdict(list)
    for name in sorted(images):
        with rasterio.open(images[name]) as image, rasterio.open(labels[name]) as label:
            if (image.count, label.count, image.width, image.height, label.width, label.height) != (3, 1, 512, 512, 512, 512):
                raise ValueError(f"Unexpected bands or 512 shape: {name}")
            if image.dtypes != ("uint8",) * 3 or label.dtypes != ("uint8",):
                raise ValueError(f"Unexpected uint8 type: {name}")
            if image.crs is None or image.crs != label.crs or image.transform != label.transform:
                raise ValueError(f"Image/label georeferencing differs: {name}")
            if CRS.from_wkt(image.crs.to_wkt()).to_epsg() != 4326:
                raise ValueError(f"Expected EPSG:4326: {name}")
            y = label.read(1)
            if not np.isin(y, (0, 1)).all():
                raise ValueError(f"Label must be binary 0/1: {name}")
            rgb = image.read()
            bounds = image.bounds
            _, _, width_m = GEOD.inv(bounds.left, (bounds.bottom+bounds.top)/2, bounds.right, (bounds.bottom+bounds.top)/2)
            _, _, height_m = GEOD.inv((bounds.left+bounds.right)/2, bounds.bottom, (bounds.left+bounds.right)/2, bounds.top)
            thumb = rgb.reshape(3, 16, 32, 16, 32).mean(axis=(2, 4))
            fingerprint = "".join(f"{v:02x}" for v in (thumb > np.median(thumb, axis=(1, 2), keepdims=True)).flatten().astype(np.uint8))
            fingerprints[fingerprint].append(name)
            rows.append({
                "id": name, "image_sha256": sha256(images[name]), "label_sha256": sha256(labels[name]),
                "positive": bool(y.any()), "target_fraction": float(y.mean()),
                "crs": "EPSG:4326", "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                "meters_per_pixel": [abs(width_m)/512, abs(height_m)/512],
                "source_scene": None, "acquisition_date": None,
            })
    image_hashes = defaultdict(list)
    for row in rows:
        image_hashes[row["image_sha256"]].append(row["id"])
    return {"schema": "pvstudy-manifest-v1", "image_dir": str(image_dir.resolve()),
            "label_dir": str(label_dir.resolve()), "images": rows,
            "exact_duplicate_images": [v for v in image_hashes.values() if len(v) > 1],
            "matching_thumbnail_fingerprints": [v for v in fingerprints.values() if len(v) > 1],
            "manual_qc_confirmed": False}


def footprint_distance_m(a: dict, b: dict) -> float:
    pa, pb = box(*a["bounds"]), box(*b["bounds"])
    p, q = nearest_points(pa, pb)
    return abs(GEOD.inv(p.x, p.y, q.x, q.y)[2])


def site_groups(rows: list[dict], distance_km: float = 5, manual_groups: list[list[str]] | None = None) -> dict[str, str]:
    parent = {r["id"]: r["id"] for r in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        x, y = find(a), find(b)
        parent[max(x, y)] = min(x, y)

    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if footprint_distance_m(a, b) < distance_km * 1000:
                union(a["id"], b["id"])
    for group in manual_groups or []:
        if not set(group) <= parent.keys():
            raise ValueError("Manual site group contains an unknown image")
        for name in group[1:]:
            union(group[0], name)
    return {name: find(name) for name in parent}
