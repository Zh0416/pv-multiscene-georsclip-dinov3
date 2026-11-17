import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
import torch
from pyproj import CRS
from rasterio.transform import from_origin
from torch import nn
from torch.nn import functional as F

from pvstudy.cache import PREPROCESS_VERSION
from pvstudy.config import Config
from pvstudy.encoders import GEOMETRIES, GeoRSCLIP, NEGATIVE_CLASSES, POSITIVE_PROMPTS
from pvstudy.io import read_json, sha256, write_json
from pvstudy.priors import area_target, fit_patch
from pvstudy.semantic_probe import _check_split, evaluate_folds, run_semantic_probe
from pvstudy.hard_negative_probe import evaluate_e03, run_hard_negative_probe
from pvstudy.l14_probe import run_l14_probe
from pvstudy.semantic_extensions import evaluate_extension, run_extension


class FakeGeo:
    open_clip_version = "fixture"
    checkpoint_sha256 = "fixture"

    def __init__(self):
        self.calls = 0

    def encode(self, rgb):
        self.calls += 1
        scores = F.avg_pool2d(rgb[:, :1].float()/255, 16) - 0.4
        return scores, torch.zeros(1)

    def encode_sliding224(self, rgb):
        self.calls += 1
        return F.avg_pool2d(rgb[:, :1].float()/255, 16) - 0.35

    def encode_semantic_pair(self, rgb, *, sliding_native=False, sliding224=False):
        self.calls += 1
        full = F.avg_pool2d(rgb[:, :1].float()/255, 16) - (0.35 if sliding_native or sliding224 else 0.4)
        return full, full + 0.1

    def encode_last_block_pair(self, rgb, *, mode="both"):
        self.calls += 1
        base = F.avg_pool2d(rgb[:, :1].float()/255, 16)
        values = {"value_only": (base - 0.3, base - 0.2),
                  "vv": (base - 0.25, base - 0.15)}
        return values if mode == "both" else {mode: values[mode]}

    def encode_window_cls(self, rgb):
        self.calls += 1
        return F.avg_pool2d(rgb[:, :1].float()/255, 16) - 0.1


class SemanticProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_sliding_stitch_preserves_patch_coordinates_and_global(self):
        geo = object.__new__(GeoRSCLIP)
        geo.device = "cpu"
        geo.model = nn.Identity().eval().requires_grad_(False)
        geo.text_positive = F.one_hot(torch.tensor(0), 512).float()
        geo.text_negative = F.one_hot(torch.tensor([1]), 512).float()
        seen_sizes = []

        def tokens(x):
            seen_sizes.append(tuple(x.shape))
            z = F.avg_pool2d(x[:, :1], 32).flatten(1)
            result = torch.zeros(x.shape[0], z.shape[1]+1, 512)
            result[:, 1:, 0] = z
            result[:, :, 1] = 1
            return result

        geo._tokens = tokens
        grid = torch.arange(256).reshape(16, 16).repeat_interleave(32, 0).repeat_interleave(32, 1)
        rgb = grid[None, None].expand(1, 3, 512, 512).to(torch.uint8)
        native, global_before = geo.encode(rgb)
        sliding = geo.encode_sliding224(rgb)
        pair_native = geo.encode_semantic_pair(rgb)
        pair_sliding = geo.encode_semantic_pair(rgb, sliding224=True)
        self.assertTrue(torch.allclose(native, sliding, atol=1e-6))
        self.assertTrue(torch.allclose(pair_native[0], native, atol=1e-6))
        self.assertTrue(torch.allclose(pair_sliding[0], sliding, atol=1e-6))
        self.assertTrue(bool((pair_native[1] > pair_native[0]).all()))
        self.assertEqual(seen_sizes[1:5], [(4, 3, 224, 224)] * 4)
        self.assertTrue(torch.isfinite(sliding).all())
        self.assertGreater(float(sliding[0,0,-1,-1]), float(sliding[0,0,0,0]))
        _, global_after = geo.encode(rgb)
        self.assertTrue(torch.equal(global_before, global_after))
        with self.assertRaisesRegex(ValueError, "one RGB512"):
            geo.encode_sliding224(rgb.expand(2, -1, -1, -1))

    def test_l14_padded_grid_and_native_windows_agree(self):
        geo = object.__new__(GeoRSCLIP)
        geo.device = "cpu"
        geo.geometry = GEOMETRIES["ViT-L-14-336"]
        geo.model = nn.Identity().eval().requires_grad_(False)
        geo.text_positive = F.one_hot(torch.tensor(0), 768).float()
        geo.text_negative = F.one_hot(torch.tensor([1]), 768).float()
        seen = []

        def tokens(x):
            seen.append(tuple(x.shape))
            patches = F.avg_pool2d(x[:, :1], 14).flatten(1)
            result = torch.zeros(x.shape[0], patches.shape[1] + 1, 768)
            result[:, 1:, 0] = patches
            result[:, :, 1] = 1
            return result

        geo._tokens = tokens
        gradient = torch.arange(512, dtype=torch.uint8)[None, None, None, :].expand(1, 3, 512, 512)
        native, _ = geo.encode(gradient)
        sliding = geo.encode_sliding_native(gradient)
        self.assertEqual(seen[0], (1, 3, 518, 518))
        self.assertEqual(len(seen), 6)
        self.assertTrue(all(shape[2:] == (336, 336) for shape in seen[1:]))
        self.assertEqual(tuple(native.shape), (1, 1, 32, 32))
        self.assertTrue(torch.allclose(native, sliding, atol=1e-6))
        with self.assertRaisesRegex(ValueError, "encode_sliding_native"):
            geo.encode_sliding224(gradient)

        class CLSModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.batches = []

            def encode_image(self, x):
                self.batches.append(tuple(x.shape))
                return F.pad(x.mean(dim=(2, 3)), (0, 765))

        geo.model = CLSModel().eval().requires_grad_(False)
        cls = geo.encode_window_cls(gradient)
        self.assertEqual(tuple(cls.shape), (1, 1, 32, 32))
        self.assertEqual(geo.model.batches, [(7, 3, 336, 336)] * 7)

    def test_e04_last_block_and_e08_window_encoder_shapes(self):
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj_weight = nn.Parameter(torch.randn(2304, 768) * 0.01)
                self.in_proj_bias = nn.Parameter(torch.zeros(2304))
                self.num_heads, self.head_dim = 12, 64
                self.logit_scale = None
                self.ln_inner = nn.Identity()
                self.head_scale = None
                self.out_proj = nn.Linear(768, 768)

        class LastBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln_1, self.ln_2 = nn.LayerNorm(768), nn.LayerNorm(768)
                self.attn = Attention()
                self.mlp = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 768))
                self.ls_1, self.ls_2 = nn.Identity(), nn.Identity()

        class Visual(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 768, 32, 32, bias=False)
                self.class_embedding = nn.Parameter(torch.randn(768) * 0.01)
                self.positional_embedding = nn.Parameter(torch.randn(50, 768) * 0.01)
                self.ln_pre, self.ln_post = nn.LayerNorm(768), nn.LayerNorm(768)
                self.proj = nn.Parameter(torch.randn(768, 512) * 0.01)
                self.transformer = nn.Module()
                self.transformer.resblocks = nn.ModuleList([nn.Identity() for _ in range(11)] + [LastBlock()])

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = Visual()
                self.cls_batches = []

            def encode_image(self, x):
                self.cls_batches.append(tuple(x.shape))
                means = x.mean(dim=(2, 3))
                return F.pad(means, (0, 509))

        torch.manual_seed(2)
        geo = object.__new__(GeoRSCLIP)
        geo.device = "cpu"
        geo.model = Model().eval().requires_grad_(False)
        geo.text_positive = F.normalize(torch.randn(512), dim=0)
        geo.text_negative = F.normalize(torch.randn(2, 512), dim=-1)
        rgb = torch.randint(0, 256, (1, 3, 512, 512), dtype=torch.uint8)
        result = geo.encode_last_block_pair(rgb)
        self.assertEqual(set(result), {"value_only", "vv"})
        for full, positive in result.values():
            self.assertEqual(tuple(full.shape), (1, 1, 32, 32))
            self.assertTrue(torch.isfinite(full).all())
            self.assertTrue(torch.isfinite(positive).all())
            self.assertLessEqual(float((positive - full).abs().max()), 1.00001)
        self.assertFalse(torch.equal(result["value_only"][0], result["vv"][0]))
        for mode in ("value_only", "vv"):
            one = geo.encode_last_block_pair(rgb, mode=mode)
            self.assertTrue(torch.equal(one[mode][0], result[mode][0]))
        cls = geo.encode_window_cls(rgb)
        self.assertEqual(tuple(cls.shape), (1, 1, 32, 32))
        self.assertTrue(torch.isfinite(cls).all())
        self.assertEqual(geo.model.cls_batches, [(7, 3, 224, 224)] * 7)
        self.assertTrue(torch.equal(cls, geo.encode_window_cls(rgb)))
        with self.assertRaisesRegex(ValueError, "one RGB512"):
            geo.encode_window_cls(rgb.expand(2, -1, -1, -1))

    def test_l14_e04_supports_openclip_multihead_attention(self):
        class LastBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln_1, self.ln_2 = nn.LayerNorm(1024), nn.LayerNorm(1024)
                self.attn = nn.MultiheadAttention(1024, 16)
                self.mlp = nn.Sequential(nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 1024))
                self.ls_1, self.ls_2 = nn.Identity(), nn.Identity()

        class Visual(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 1024, 14, 14, bias=False)
                self.class_embedding = nn.Parameter(torch.zeros(1024))
                self.positional_embedding = nn.Parameter(torch.randn(577, 1024) * 0.01)
                self.ln_pre, self.ln_post = nn.LayerNorm(1024), nn.LayerNorm(1024)
                self.proj = nn.Parameter(torch.randn(1024, 768) * 0.01)
                self.transformer = nn.Module()
                self.transformer.resblocks = nn.ModuleList([nn.Identity() for _ in range(23)] + [LastBlock()])

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = Visual()

        geo = object.__new__(GeoRSCLIP)
        geo.device = "cpu"
        geo.geometry = GEOMETRIES["ViT-L-14-336"]
        geo._sequence_first = True
        geo.model = Model().eval().requires_grad_(False)
        geo.text_positive = F.normalize(torch.randn(768), dim=0)
        geo.text_negative = F.normalize(torch.randn(2, 768), dim=-1)
        result = geo.encode_last_block_pair(torch.zeros(1, 3, 512, 512, dtype=torch.uint8))
        self.assertEqual(set(result), {"value_only", "vv"})
        for full, positive in result.values():
            self.assertEqual(tuple(full.shape), (1, 1, 32, 32))
            self.assertTrue(torch.isfinite(full).all() and torch.isfinite(positive).all())

    def _records(self):
        records = []
        assignment = {}
        for i in range(8):
            mask = np.zeros((512,512), np.float32)
            if i % 2 == 0:
                mask[:, :256] = 1
            target = area_target(mask)
            records.append({"id": f"{i}.tif", "site": f"site-{i}", "mask": mask, "y32": target,
                            "native512": target*0.8-0.4, "sliding224": target*0.7-0.3})
            assignment[f"{i}.tif"] = {"site": f"site-{i}", "partition": "dev", "fold": i//2}
        assignment["sealed.tif"] = {"site": "test-site", "partition": "test", "fold": None}
        return records, {"status": "locked_after_manual_qc", "qc_sha256": "qc", "assignment": assignment}

    def test_calibration_uses_only_outer_training_sites(self):
        records, split = self._records()
        calls = []

        def spy(train, key):
            calls.append(({r["id"] for r in train}, key))
            return fit_patch(train, key)

        with patch("pvstudy.semantic_probe.fit_patch", side_effect=spy):
            summary, folds, predictions = evaluate_folds(records, split)
        self.assertEqual(len(calls), 8)
        for fold in folds:
            expected = {i for i,v in split["assignment"].items() if v["partition"] == "dev" and v["fold"] != fold["fold"]}
            for ids, key in calls[2*fold["fold"]:2*fold["fold"]+2]:
                self.assertEqual(ids, expected)
                self.assertTrue(ids.isdisjoint(fold["validation_ids"]))
            self.assertEqual(set(predictions["sliding224"]), {r["id"] for r in records})
        self.assertTrue(summary["E01_semantic"]["sliding224"]["quality_pass"])
        with self.assertRaisesRegex(ValueError, "exactly the development"):
            evaluate_folds(records + [{**records[0], "id": "sealed.tif"}], split)

    def test_split_requires_qc_and_disjoint_sites(self):
        records, split = self._records()
        manifest = {"manual_qc_confirmed": True, "manual_qc_sha256": "qc",
                    "images": [{"id": i} for i in split["assignment"]]}
        self.assertEqual(len(_check_split(manifest, split)), 8)
        with self.assertRaises(PermissionError):
            _check_split({**manifest, "manual_qc_confirmed": False}, split)
        bad = copy.deepcopy(split)
        bad["assignment"]["sealed.tif"]["site"] = "site-0"
        with self.assertRaisesRegex(ValueError, "Site crosses"):
            _check_split(manifest, bad)

    def test_e03_recalibrates_each_text_mode_inside_outer_training_fold(self):
        records, split = self._records()
        for r in records:
            r["native_full"] = r["native512"]
            r["native_positive_only"] = r["native512"] + 0.1
            r["sliding_full"] = r["sliding224"]
            r["sliding_positive_only"] = r["sliding224"] + 0.1
        calls = []

        def spy(train, key):
            calls.append(({r["id"] for r in train}, key))
            return fit_patch(train, key)

        with patch("pvstudy.hard_negative_probe.fit_patch", side_effect=spy):
            summary, folds, predictions = evaluate_e03(records, split)
        self.assertEqual(len(calls), 16)
        self.assertEqual(set(predictions), {"constant", "native_full", "native_positive_only", "sliding_full", "sliding_positive_only"})
        for fold in folds:
            expected = set(fold["train_ids"])
            for ids, mode in calls[4*fold["fold"]:4*fold["fold"]+4]:
                self.assertEqual(ids, expected)
                self.assertTrue(ids.isdisjoint(fold["validation_ids"]))
            self.assertEqual(len(fold["calibrators"]), 4)
        self.assertIn("native", summary["E03"])
        with self.assertRaisesRegex(ValueError, "exactly the locked development"):
            evaluate_e03(records + [{**records[0], "id": "sealed.tif"}], split)

    def test_extension_calibration_uses_only_outer_training_sites(self):
        records, split = self._records()
        for r in records:
            r["native_full"] = r["native512"]
            r["native_positive_only"] = r["native512"] + 0.1
            r["value_only_full"] = r["native512"] + 0.15
            r["vv_full"] = r["native512"] + 0.2
            r["value_only_positive_only"] = r["native512"] + 0.25
            r["vv_positive_only"] = r["native512"] + 0.3
        modes = ["constant", "native_full", "value_only_full", "vv_full",
                 "native_positive_only", "value_only_positive_only", "vv_positive_only"]
        calls = []

        def spy(train, key):
            calls.append(({r["id"] for r in train}, key))
            return fit_patch(train, key)

        with patch("pvstudy.semantic_extensions.fit_patch", side_effect=spy):
            summary, folds, predictions = evaluate_extension(records, split, modes)
        self.assertEqual(len(calls), 24)
        for fold in folds:
            start = 6 * fold["fold"]
            for ids, key in calls[start:start + 6]:
                self.assertEqual(ids, set(fold["train_ids"]))
                self.assertTrue(ids.isdisjoint(fold["validation_ids"]))
            self.assertEqual(len(fold["calibrators"]), 6)
        self.assertEqual(set(predictions["vv_full"]), {r["id"] for r in records})
        self.assertIn("E04_replacement_pass", summary["comparisons"]["vv_full"])
        with self.assertRaisesRegex(ValueError, "exactly the locked development"):
            evaluate_extension(records + [{**records[0], "id": "sealed.tif"}], split, modes)

    def test_l14_diagnostic_isolated_and_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir, label_dir, output = root / "images", root / "labels", root / "l14-output"
            for path in (image_dir, label_dir, output):
                path.mkdir()
            records, split = self._records()
            rows = []
            for record in records:
                image_id = record["id"]
                rgb = np.repeat((record["mask"] * 255).astype(np.uint8)[None], 3, axis=0)
                for path, data in ((image_dir / image_id, rgb),
                                   (label_dir / image_id, record["mask"][None].astype(np.uint8))):
                    with rasterio.open(path, "w", driver="GTiff", height=512, width=512,
                                       count=data.shape[0], dtype="uint8",
                                       transform=from_origin(112, 22, 0.001, 0.001),
                                       crs=CRS.from_epsg(4326).to_wkt()) as target:
                        target.write(data)
                rows.append({"id": image_id, "image_sha256": sha256(image_dir / image_id),
                             "label_sha256": sha256(label_dir / image_id),
                             "target_fraction": float(record["mask"].mean())})
            rows.append({"id": "sealed.tif", "image_sha256": "unread", "label_sha256": "unread",
                         "target_fraction": 0})
            write_json(output / "dataset_manifest.json", {
                "manual_qc_confirmed": True, "manual_qc_sha256": "qc", "images": rows})
            write_json(output / "split_lock.json", split)
            config = Config(image_dir, label_dir, root / "geo.pt", root / "dino.pth",
                            root / "source", output, device="cpu", georsclip_sha256="fixture",
                            georsclip_model="ViT-L-14-336")
            fake = FakeGeo()
            with patch("pvstudy.l14_probe.GeoRSCLIP", return_value=fake):
                summary = run_l14_probe(config)
                calls = fake.calls
                self.assertEqual(summary, run_l14_probe(config))
                self.assertEqual(fake.calls, calls)
            self.assertEqual(summary["development_images"], 8)
            self.assertIn("window_cls_vs_e05_positive_only", summary["comparisons"])
            self.assertTrue((output / "diagnostics/l14-semantic-v1/l14_review.zip").is_file())
            self.assertFalse((output / "cache").exists())
            meta_path = next((output / "diagnostics/l14-semantic-v1/raw").glob("*.json"))
            altered = read_json(meta_path)
            altered["sha256"] = "wrong"
            write_json(meta_path, altered)
            with patch("pvstudy.l14_probe.GeoRSCLIP", return_value=fake):
                with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                    run_l14_probe(config)

    def test_isolated_probe_resume_and_input_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir, label_dir, output = root/"images", root/"labels", root/"output"
            for p in (image_dir, label_dir, output/"cache"):
                p.mkdir(parents=True)
            records, split = self._records()
            rows, entries = [], {}
            fake = FakeGeo()
            for i, record in enumerate(records):
                image_id = record["id"]
                rgb = np.repeat((record["mask"]*255).astype(np.uint8)[None], 3, axis=0)
                for path, data in ((image_dir/image_id, rgb), (label_dir/image_id, record["mask"][None].astype(np.uint8))):
                    with rasterio.open(path, "w", driver="GTiff", height=512, width=512,
                                       count=data.shape[0], dtype="uint8", transform=from_origin(112,22,0.001,0.001), crs=CRS.from_epsg(4326).to_wkt()) as target:
                        target.write(data)
                native, glob = fake.encode(torch.from_numpy(rgb)[None])
                cache_path = output/"cache"/f"{i}.npz"
                np.savez_compressed(cache_path, semantic_raw=native[0,0].numpy(), global_raw=np.float32(glob[0]))
                row = {"id": image_id, "image_sha256": sha256(image_dir/image_id),
                       "label_sha256": sha256(label_dir/image_id), "target_fraction": float(record["mask"].mean())}
                rows.append(row)
                entries[image_id] = {"file":cache_path.name, "cache_sha256":sha256(cache_path), "image_sha256":row["image_sha256"]}
            # No test image/label exists: accessing a test file would fail this integration test.
            rows.append({"id":"sealed.tif", "image_sha256":"unread", "label_sha256":"unread", "target_fraction":0})
            manifest = {"manual_qc_confirmed":True, "manual_qc_sha256":"qc", "images":rows}
            write_json(output/"dataset_manifest.json", manifest)
            write_json(output/"split_lock.json", split)
            write_json(output/"cache/index.json", {"identity": {"preprocess":PREPROCESS_VERSION,
                "augmentation":"none", "georsclip_sha256":"fixture", "open_clip_version":"fixture",
                "prompts_positive":list(POSITIVE_PROMPTS), "prompts_negative":list(NEGATIVE_CLASSES),
                "georsclip_dense":"native patch tokens, bicubic 7x7 positional interpolation to 16x16"}, "entries":entries})
            original_records = [{**r, "native512":r["y32"]-0.4, "sliding224":r["y32"]-0.35} for r in records]
            _, native_folds, _ = evaluate_folds(original_records, split)
            for fold in native_folds:
                write_json(output/f"fold-{fold['fold']}/seed-42/result-cam.json", {
                    "train_ids":fold["train_ids"], "validation_ids":fold["validation_ids"],
                    "cam":{"C1":fold["metrics"]["native512"]}})
            config = Config(image_dir, label_dir, root/"geo.pt", root/"dino.pth", root/"source", output,
                            device="cpu", georsclip_sha256="fixture")
            originals = {p:sha256(p) for p in output.rglob("*") if p.is_file()}
            with patch("pvstudy.semantic_probe.GeoRSCLIP", return_value=fake):
                summary = run_semantic_probe(config)
                calls = fake.calls
                repeated = run_semantic_probe(config)
                self.assertEqual(calls, fake.calls)
                self.assertEqual(summary, repeated)
                self.assertEqual(summary["development_images"], 8)
                self.assertEqual(summary["earlier_C1_reproduced_folds"], [0,1,2,3])
                self.assertFalse(summary["E05_replacement"]["gpu_latency_measured"])
                self.assertTrue((output/"diagnostics/e05-native224-v1/e05_review.zip").is_file())
                for p, digest in originals.items():
                    self.assertEqual(sha256(p), digest)
                # Recovery uses original JSON references, with no feature arrays left.
                for p in (output/"cache").glob("*.npz"):
                    p.unlink()
                recovered = run_semantic_probe(config, run_name="recovered", recompute_native=True)
                self.assertEqual(recovered["aggregate"], summary["aggregate"])
                self.assertFalse(recovered["original_native_arrays_verified"])
                self.assertEqual(recovered["earlier_C1_reproduced_folds"], [0,1,2,3])
                calls = fake.calls
                self.assertEqual(recovered, run_semantic_probe(config, run_name="recovered", recompute_native=True))
                self.assertEqual(calls, fake.calls)
                self.assertFalse(list((output/"cache").glob("*.npz")))
                with patch("pvstudy.hard_negative_probe.GeoRSCLIP", return_value=fake):
                    e03 = run_hard_negative_probe(config, e05_run_name="recovered")
                    calls = fake.calls
                    self.assertEqual(e03, run_hard_negative_probe(config, e05_run_name="recovered"))
                    self.assertEqual(calls, fake.calls)
                    self.assertEqual(e03["original_C1_reproduced_folds"], [0,1,2,3])
                    self.assertEqual(e03["E05_full_text_reproduced_folds"], [0,1,2,3])
                    self.assertTrue((output/"diagnostics/e03-no-hard-negatives-v1/e03_review.zip").is_file())
                    with patch("pvstudy.semantic_extensions.GeoRSCLIP", return_value=fake):
                        for method in ("e04", "e08"):
                            extension = run_extension(config, method=method, e03_run_name="e03-no-hard-negatives-v1")
                            calls = fake.calls
                            self.assertEqual(extension, run_extension(config, method=method))
                            self.assertEqual(calls, fake.calls)
                            self.assertEqual(extension["reference_reproduced_folds"], [0,1,2,3])
                            self.assertIn("sliding_full", extension["E03_E05_reference_aggregate"])
                            if method == "e08":
                                self.assertIn("window_cls_vs_e05_positive_only", extension["comparisons"])
                            else:
                                self.assertEqual(set(extension["timing"]), {"value_only", "vv", "native_cls_check"})
                            name = "e04-last-block-v1" if method == "e04" else "e08-window-cls-v1"
                            self.assertTrue((output/"diagnostics"/name/f"{method}_review.zip").is_file())
                            self.assertTrue(list((output/"diagnostics"/name/"predictions").rglob("*.npz")))
                        extension_meta = next((output/"diagnostics/e04-last-block-v1/raw").glob("*.json"))
                        saved = extension_meta.read_bytes()
                        altered = read_json(extension_meta)
                        altered["sha256"] = "tampered"
                        write_json(extension_meta, altered)
                        with self.assertRaisesRegex(ValueError, "Extension raw-map integrity mismatch"):
                            run_extension(config, method="e04")
                        extension_meta.write_bytes(saved)
                    for p, digest in originals.items():
                        if p.exists():
                            self.assertEqual(sha256(p), digest)
                    e05_meta_path = next((output/"diagnostics/recovered/raw").glob("*.json"))
                    e05_meta_original = e05_meta_path.read_bytes()
                    e05_meta = read_json(e05_meta_path)
                    e05_meta["sha256"] = "tampered"
                    write_json(e05_meta_path, e05_meta)
                    with self.assertRaisesRegex(ValueError, "E05 raw map integrity mismatch"):
                        run_hard_negative_probe(config, e05_run_name="recovered")
                    e05_meta_path.write_bytes(e05_meta_original)
                reference = output/"fold-0/seed-42/result-cam.json"
                original_reference = reference.read_bytes()
                altered = read_json(reference)
                altered["cam"]["C1"]["site_rows"][0]["ap"] = 0.123
                write_json(reference, altered)
                with self.assertRaisesRegex(ValueError, "Recomputed C1 differs"):
                    run_semantic_probe(config, run_name="bad-reference", recompute_native=True)
                reference.write_bytes(original_reference)
                metadata_path = next((output/"diagnostics/recovered/raw").glob("*.json"))
                metadata = read_json(metadata_path)
                metadata["sha256"] = "tampered"
                write_json(metadata_path, metadata)
                with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                    run_semantic_probe(config, run_name="recovered", recompute_native=True)


if __name__ == "__main__":
    unittest.main()
