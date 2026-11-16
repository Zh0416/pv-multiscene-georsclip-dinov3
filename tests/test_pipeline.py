import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin

from pvstudy.cache import PREPROCESS_VERSION
from pvstudy.config import Config
from pvstudy.data import build_manifest
from pvstudy.experiment import run_fold
from pvstudy.io import sha256, write_json


class PipelineTests(unittest.TestCase):
    def test_locked_synthetic_fold_and_cam_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, labels, cache = root / "images", root / "labels", root / "outputs" / "cache"
            images.mkdir()
            labels.mkdir()
            cache.mkdir(parents=True)
            transform = from_origin(112, 22, 0.00002, 0.00002)
            rng = np.random.default_rng(11)
            entries = {}
            for i in range(8):
                image_id = f"site-{i}.tif"
                positive = i < 4
                mask = np.zeros((512, 512), dtype=np.uint8)
                if positive:
                    mask[:, :256] = 1
                rgb = np.full((3, 512, 512), 40 + i, dtype=np.uint8)
                for path, array in ((images / image_id, rgb), (labels / image_id, mask[None])):
                    with rasterio.open(path, "w", driver="GTiff", height=512, width=512,
                                       count=array.shape[0], dtype="uint8", crs=CRS.from_epsg(4326).to_wkt(),
                                       transform=transform) as target:
                        target.write(array)
                dino = rng.normal(0, 0.01, (1024, 32, 32)).astype(np.float32)
                if positive:
                    dino[0, :, :16] += 3
                semantic = (mask.reshape(32, 16, 32, 16).mean((1, 3)) * 0.8 +
                            rng.normal(0, 0.02, (32, 32))).astype(np.float32)
                feature_path = cache / f"{i}.npz"
                np.savez_compressed(feature_path, dino=dino, semantic_raw=semantic,
                                    global_raw=np.float32(positive))
                entries[image_id] = {"file": feature_path.name, "cache_sha256": sha256(feature_path),
                                     "image_sha256": sha256(images / image_id)}
            manifest = build_manifest(images, labels)
            assignment = {f"site-{i}.tif": {"site": str(i), "partition": "dev", "fold": i % 4}
                          for i in range(8)}
            split = {"status": "locked_after_manual_qc", "assignment": assignment}
            config = Config(images, labels, root / "geo.pt", root / "dino.pth", root / "source",
                            root / "outputs", device="cpu", georsclip_sha256="synthetic",
                            dinov3_sha256_prefix="eadcf0ff")
            write_json(cache / "index.json", {"identity": {"preprocess": PREPROCESS_VERSION,
                       "augmentation": "none", "georsclip_sha256": "synthetic",
                       "dinov3_sha256": "eadcf0ff123"}, "entries": entries})
            cam = run_fold(config, manifest, split, fold=0, seed=42, variant=None, epochs=1, k=1)
            self.assertIn("C6", cam["cam"])
            self.assertNotIn("C0b", cam["cam"])
            ready = root / "outputs" / "fold-0" / "seed-42" / "cam_ready.json"
            original_hash = sha256(ready)
            result = run_fold(config, manifest, split, fold=0, seed=42, variant="S2", epochs=1, k=1)
            self.assertEqual(original_hash, sha256(ready))
            self.assertIsNotNone(result["segmentation"])
            self.assertEqual(len(result["segmentation"]["prediction_files"]), 2)


if __name__ == "__main__":
    unittest.main()
