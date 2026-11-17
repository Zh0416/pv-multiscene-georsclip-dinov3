import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
import torch
from pyproj import CRS
from rasterio.transform import from_origin

from pvstudy.cache import _name, build_cache, load_feature, verify_e08_cache
from pvstudy.cache import PREPROCESS_VERSION
from pvstudy.config import Config
from pvstudy.encoders import GEOMETRIES, NEGATIVE_CLASSES, POSITIVE_PROMPTS
from pvstudy.experiment import run_fold
from pvstudy.io import read_json, sha256, write_json
from pvstudy.l14_probe import PROTOCOL
from pvstudy.priors import area_target
from pvstudy.splits import partition_ids


class FakeGeo:
    checkpoint_sha256 = "fixture"
    open_clip_version = "fixture"
    geometry = GEOMETRIES["ViT-L-14-336"]

    def __init__(self):
        self.calls = 0

    def encode(self, rgb):
        self.calls += 1
        return torch.zeros((1, 1, 32, 32)), torch.zeros(1)


class FakeDino:
    checkpoint_sha256 = "eadcf0ff-fixture"
    layer_indices = (20, 21, 22, 23)

    def encode(self, rgb):
        feature = torch.zeros((1, 1024, 32, 32))
        feature[:, 0] = torch.nn.functional.avg_pool2d(rgb[:, :1].float() / 255, 16)[:, 0]
        return feature


class E08CacheTests(unittest.TestCase):
    def _fixture(self, root: Path):
        images, labels, output, source = (root / part for part in ("images", "labels", "e08", "diagnostic"))
        for path in (images, labels, output, source / "raw", root / "dinov3" / "dinov3"):
            path.mkdir(parents=True, exist_ok=True)
        (root / "dinov3" / "dinov3" / "backbones.py").write_text("# fixture\n", encoding="utf-8")
        rows, assignments, expected = [], {}, {}
        for i in range(8):
            image_id = f"dev-{i}.tif"
            mask = np.zeros((512, 512), dtype=np.uint8)
            if i < 4:
                mask[:, :256] = 1
            rgb = np.repeat((mask * 255)[None], 3, axis=0)
            for path, array in ((images / image_id, rgb), (labels / image_id, mask[None])):
                with rasterio.open(path, "w", driver="GTiff", height=512, width=512,
                                   count=array.shape[0], dtype="uint8",
                                   transform=from_origin(112, 22, 0.001, 0.001),
                                   crs=CRS.from_epsg(4326).to_wkt()) as target:
                    target.write(array)
            rows.append({"id": image_id, "image_sha256": sha256(images / image_id),
                         "label_sha256": sha256(labels / image_id)})
            assignments[image_id] = {"site": str(i), "partition": "dev", "fold": i % 4}
            raw = area_target(mask.astype(np.float32)) * 0.7 + 0.05
            expected[image_id] = raw
            stem = Path(_name(image_id)).stem
            raw_path = source / "raw" / f"{stem}.npz"
            np.savez_compressed(raw_path, window_cls_positive_only=raw, global_raw=np.float32(0))
            write_json(source / "raw" / f"{stem}.json", {
                "image_id": image_id, "sha256": sha256(raw_path), "native_global_unchanged": True})
        rows.append({"id": "sealed.tif", "image_sha256": "unread", "label_sha256": "unread"})
        assignments["sealed.tif"] = {"site": "test", "partition": "test", "fold": None}
        manifest = {"manual_qc_confirmed": True, "manual_qc_sha256": "qc", "images": rows}
        split = {"status": "locked_after_manual_qc", "qc_sha256": "qc", "assignment": assignments}
        write_json(output / "dataset_manifest.json", manifest)
        write_json(output / "split_lock.json", split)
        code = {name: sha256(Path(__file__).resolve().parents[1] / "pvstudy" / name)
                for name in ("l14_probe.py", "encoders.py", "semantic_extensions.py", "priors.py", "metrics.py")}
        write_json(source / "identity.json", {
            "protocol": PROTOCOL, "georsclip_model": "ViT-L-14-336",
            "georsclip_sha256": "fixture", "open_clip_version": "fixture",
            "preprocess": PREPROCESS_VERSION, "augmentation": "none",
            "prompts_positive": list(POSITIVE_PROMPTS), "prompts_negative": list(NEGATIVE_CLASSES),
            "manifest_sha256": sha256(output / "dataset_manifest.json"),
            "split_sha256": sha256(output / "split_lock.json"), "code_sha256": code})
        write_json(source / "l14_summary.json", {
            "status": "development_l14_diagnostic_complete", "development_images": 8,
            "identity_sha256": sha256(source / "identity.json")})
        config = Config(images, labels, root / "geo.pt", root / "dino.pth", root / "dinov3",
                        output, device="cpu", georsclip_sha256="fixture",
                        georsclip_model="ViT-L-14-336", semantic_mode="window_cls_positive_only",
                        semantic_diagnostic_dir=source,
                        semantic_summary_sha256=sha256(source / "l14_summary.json"),
                        dinov3_source_commit="fixture")
        return config, manifest, split, expected

    def test_approved_raw_cache_resumes_and_feeds_fold(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, manifest, split, expected = self._fixture(Path(temporary))
            ids = partition_ids(split, "dev")
            geo = FakeGeo()
            with patch("pvstudy.cache.load_encoders", return_value=(geo, FakeDino())):
                cache = build_cache(config, manifest, ids)
                self.assertEqual(geo.calls, 8)
                self.assertEqual(build_cache(config, manifest, ids), cache)
                self.assertEqual(geo.calls, 8)
            for image_id in ids:
                item = load_feature(config, manifest, cache, image_id)
                np.testing.assert_array_equal(item["semantic_raw"], expected[image_id])
            report = verify_e08_cache(config)
            self.assertEqual(report["development_images"], 8)
            self.assertEqual(report["test_images_read"], 0)
            self.assertEqual(report["semantic_mode"], "window_cls_positive_only")
            self.assertEqual(report["cache_index_sha256"], sha256(config.output_dir / "cache" / "index.json"))
            fold = run_fold(config, manifest, split, fold=0, seed=42, variant=None,
                            epochs=1, k=1)
            self.assertEqual(fold["cam_identity"],
                             read_json(config.output_dir / "fold-0" / "seed-42" / "cam_ready.json")["identity"])
            self.assertEqual(read_json(config.output_dir / "cache" / "index.json")["identity"]["semantic_mode"],
                             "window_cls_positive_only")
            split_path = config.output_dir / "split_lock.json"
            altered_split = read_json(split_path)
            altered_split["unexpected_change"] = True
            write_json(split_path, altered_split)
            with self.assertRaisesRegex(ValueError, "verification is stale"):
                run_fold(config, manifest, split, fold=0, seed=42, variant=None, epochs=1, k=1)
            write_json(split_path, split)
            index_path = config.output_dir / "cache" / "index.json"
            altered_index = read_json(index_path)
            altered_index["unexpected_change"] = True
            write_json(index_path, altered_index)
            with self.assertRaisesRegex(ValueError, "verification is stale"):
                run_fold(config, manifest, split, fold=0, seed=42, variant=None, epochs=1, k=1)

    def test_rejects_unverified_or_tampered_e08_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, manifest, split, _ = self._fixture(Path(temporary))
            ids = partition_ids(split, "dev")
            with self.assertRaisesRegex(FileNotFoundError, "verify-e08-cache"):
                run_fold(config, manifest, split, fold=0, seed=42, variant=None, epochs=1, k=1)
            with self.assertRaisesRegex(PermissionError, "test stays sealed"):
                build_cache(config, manifest, ids + ["sealed.tif"])
            source = config.semantic_diagnostic_dir
            raw_path = source / "raw" / _name(ids[0])
            with np.load(raw_path, allow_pickle=False) as archive:
                original = archive["window_cls_positive_only"].copy()
            np.savez_compressed(raw_path, window_cls_positive_only=original + 0.1,
                                global_raw=np.float32(0))
            with patch("pvstudy.cache.load_encoders", return_value=(FakeGeo(), FakeDino())):
                with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                    build_cache(config, manifest, ids)
            altered = replace(config, semantic_summary_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "approved run"):
                build_cache(altered, manifest, ids)


if __name__ == "__main__":
    unittest.main()
