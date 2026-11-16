import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from pyproj import CRS
import rasterio
from rasterio.transform import from_origin

from pvstudy.io import sha256
from pvstudy.qc import export_qc, review_identity, validate_qc


class QualityReviewTests(unittest.TestCase):
    def fixtures(self, root):
        images, labels = root / "images", root / "labels"
        images.mkdir()
        labels.mkdir()
        name = "sample.tif"
        rgb = np.full((3, 512, 512), 70, dtype=np.uint8)
        mask = np.zeros((1, 512, 512), dtype=np.uint8)
        mask[:, :20, :20] = 1
        for folder, pixels in ((images, rgb), (labels, mask)):
            with rasterio.open(folder / name, "w", driver="GTiff", count=pixels.shape[0],
                               width=512, height=512, dtype="uint8",
                               crs=CRS.from_epsg(4326).to_wkt(),
                               transform=from_origin(112, 22, 0.00002, 0.00002)) as dst:
                dst.write(pixels)
        manifest = {"images": [{"id": name, "positive": True, "target_fraction": 400/512**2,
                                "bounds": [112, 21.99, 112.01, 22],
                                "image_sha256": sha256(images / name),
                                "label_sha256": sha256(labels / name)}],
                    "exact_duplicate_images": [], "matching_thumbnail_fingerprints": []}
        split = {"status": "candidate_requires_manual_qc",
                 "assignment": {name: {"site": name, "partition": "dev", "fold": 0}}}
        return images, labels, manifest, split

    def test_qc_export_never_confirms_and_checks_hashes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, labels, manifest, split = self.fixtures(root)
            original = copy.deepcopy((manifest, split))
            page = export_qc(images, labels, manifest, split, root / "qc")
            self.assertIn("data:image/png;base64,", page.read_text(encoding="utf-8"))
            draft = json.loads((root / "qc" / "QC.template.json").read_text(encoding="utf-8"))
            self.assertEqual(draft["reviewed_ids"], [])
            self.assertFalse(draft["source_scene_checked"])
            self.assertEqual((manifest, split), original)
            with self.assertRaises(ValueError):
                validate_qc(manifest, split, draft)
            manifest["images"][0]["image_sha256"] = "0"*64
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                export_qc(images, labels, manifest, split, root / "bad")

    def test_qc_rejects_stale_split_unresolved_issues_and_truthy_strings(self):
        manifest = {"images": [{"id": "a"}], "exact_duplicate_images": [],
                    "matching_thumbnail_fingerprints": []}
        split = {"assignment": {"a": {"site": "one", "fold": 0, "partition": "dev"}}}
        qc = {"schema": "pvstudy-manual-qc-v1", "review_identity": review_identity(manifest, split),
              "reviewed_ids": ["a"], "band_order_confirmed": True,
              "label_alignment_confirmed": True, "source_scene_checked": True,
              "image_reviews": {"a": {"status": "reviewed", "notes": ""}}}
        validate_qc(manifest, split, qc)
        for bad in ({**qc, "source_scene_checked": "false"},
                    {**qc, "pending_issues": [{"id": "a"}]},
                    {**qc, "reviewed_ids": ["a", "a"]},
                    {**qc, "image_reviews": {"a": {"status": "issue"}}},
                    {**qc, "review_identity": "stale"}):
            with self.assertRaises(ValueError):
                validate_qc(manifest, split, bad)
        split["assignment"]["a"]["fold"] = 1
        with self.assertRaisesRegex(ValueError, "different manifest or split"):
            validate_qc(manifest, split, qc)


if __name__ == "__main__":
    unittest.main()
