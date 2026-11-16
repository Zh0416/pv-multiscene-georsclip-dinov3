import unittest
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import numpy as np
import torch

from pvstudy.metrics import average_precision, paired_site_bootstrap, site_metrics
from pvstudy.experiment import _load_prior_maps, _save_prior_maps, train_decoder, train_gate
from pvstudy.models import Decoder, FusionGate, local_indices
from pvstudy.priors import area_target, fit_calibrator, fit_prior, site_weights
from pvstudy.splits import partition_ids
from pvstudy.source import prepare_dinov3_source
from pvstudy.config import Config


class ProtocolTests(unittest.TestCase):
    def test_area_target_and_calibration(self):
        mask = np.zeros((512, 512), dtype=np.float32)
        mask[:8, :16] = 1
        self.assertAlmostEqual(float(area_target(mask)[0, 0]), 0.5)
        records = [{"site": "a"}, {"site": "a"}, {"site": "b"}]
        np.testing.assert_allclose(site_weights(records, pixels=False), [0.25, 0.25, 0.5])
        score = np.array([-0.5, -0.1, 0.1, 0.5])
        target = np.array([0, 0.1, 0.9, 1])
        fit = fit_calibrator(score, target, np.ones(4) / 4)
        self.assertTrue(fit.success)
        self.assertTrue(np.all(np.diff(fit.predict(score)) > 0))
        self.assertTrue(0.02 <= fit.temperature <= 1)

    def test_split_test_sealed(self):
        split = {"status": "candidate_requires_manual_qc", "assignment": {
            "a": {"site": "one", "partition": "dev", "fold": 0},
            "b": {"site": "two", "partition": "dev", "fold": 1},
            "c": {"site": "three", "partition": "test", "fold": None}}}
        self.assertEqual(partition_ids(split, "train", 0), ["b"])
        with self.assertRaises(PermissionError):
            partition_ids(split, "test", unlock_test=True)

    def test_existing_source_must_match_pinned_commit(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "dinov3-source"
            (source / "dinov3" / "hub").mkdir(parents=True)
            (source / "dinov3" / "hub" / "backbones.py").write_text("# fixture\n", encoding="utf-8")
            subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(source), "-c", "user.name=Test",
                            "-c", "user.email=test@example.invalid", "commit", "-m", "fixture"],
                           check=True, capture_output=True)
            commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            self.assertEqual(prepare_dinov3_source(source, commit)["commit"], commit)
            with self.assertRaises(ValueError):
                prepare_dinov3_source(source, "0" * 40)

    def test_kaggle_mount_config(self):
        path = Path(__file__).resolve().parents[1] / "config.kaggle.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(raw["label_dir"], raw["image_dir"].rsplit("/", 1)[0] + "/labels")
        self.assertEqual(raw["dinov3_source_commit"], "6876159a11b4df116f30f667f8c9888617df0751")
        if os.name == "nt":
            with self.assertRaisesRegex(ValueError, "Linux Kaggle runtime"):
                Config.load(path)
        else:
            self.assertEqual(str(Config.load(path).image_dir), raw["image_dir"])

    def test_metrics_ties_and_pairing(self):
        self.assertAlmostEqual(average_precision(np.array([1, 0, 1, 0]), np.ones(4)), 0.5)
        paired = paired_site_bootstrap({"a": 0.8, "b": 0.7}, {"a": 0.6, "b": 0.6}, draws=100)
        self.assertAlmostEqual(paired["mean_delta"], 0.15)
        self.assertEqual(paired["sites"], 2)

    def test_cam_ap_pools_site_patches(self):
        mask = np.ones((512, 512), dtype=np.float32)
        first = {"id": "a", "site": "one", "mask": mask,
                 "y32": np.array([[1, 0]], dtype=np.float32)}
        second = {"id": "b", "site": "one", "mask": mask,
                  "y32": np.array([[1, 0]], dtype=np.float32)}
        predictions = {"a": np.array([[0.9, 0.8]]), "b": np.array([[0.2, 0.1]])}
        result = site_metrics([first, second], predictions, cam=True)
        pooled = average_precision(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.2, 0.1]))
        self.assertAlmostEqual(result["site_macro_ap"], pooled)

    def test_prior_map_integrity_roundtrip(self):
        record = {"id": "nested/a.tif", "mask": np.zeros((512, 512), dtype=np.float32)}
        for key in ("p_sem", "p_spa", "p_bank", "p_self", "C5", "C5g", "C6", "C7", "a", "rho", "R"):
            record[key] = np.full((32, 32), 0.3, dtype=np.float32)
        record["route"] = np.full((3, 32, 32), 1/3, dtype=np.float32)
        record.update({"s_global": 0.2, "r_seed": 0.1, "seed_count": 2})
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            saved = _save_prior_maps(folder, [record])
            loaded = _load_prior_maps(folder, [{"id": record["id"], "mask": record["mask"]}], saved)
            np.testing.assert_array_equal(loaded[0]["C6"], record["C6"])
            self.assertEqual(loaded[0]["y32"].shape, (32, 32))
            bad = {record["id"]: {**saved[record["id"]], "sha256": "bad"}}
            with self.assertRaises(ValueError):
                _load_prior_maps(folder, [{"id": record["id"], "mask": record["mask"]}], bad)

    def test_attention_indices_and_gate(self):
        indices, valid = local_indices()
        self.assertEqual(indices.shape, (4096, 9))
        self.assertEqual(int(valid[0].sum()), 4)
        self.assertEqual(int(indices[0, 4]), 0)
        self.assertEqual(int(indices[2, 4]), 1)
        self.assertEqual(int(valid[32 * 64 + 32].sum()), 9)
        gate = FusionGate("C6", theta=0.4)
        dino = torch.randn(1, 1024, 32, 32)
        p_sem = torch.rand(1, 1, 32, 32)
        p_spa = torch.rand_like(p_sem)
        with torch.no_grad():
            m, details = gate(dino, p_sem, p_spa, torch.tensor([0.5]))
        self.assertTrue(bool(((m >= 0) & (m <= 1)).all()))
        self.assertTrue(torch.allclose(details["route"].sum(1), torch.ones_like(m[:, 0]), atol=1e-6))
        self.assertGreaterEqual(float(details["R"].min()), 0.15)

    def test_decoder_shape_and_bias_gradient(self):
        torch.set_num_threads(2)
        model = Decoder()
        rgb = torch.rand(1, 3, 512, 512)
        dino = torch.randn(1, 1024, 32, 32)
        m = torch.rand(1, 1, 32, 32)
        with torch.no_grad():
            logits = model(rgb, dino, m, bias_enabled=True)
        self.assertEqual(tuple(logits.shape), (1, 1, 512, 512))
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_inner_bank_uses_other_sites(self):
        rng = np.random.default_rng(8)
        records = []
        for i in range(8):
            positive = i < 4
            mask = np.zeros((512, 512), dtype=np.float32)
            if positive:
                mask[:, :256] = 1
            truth = area_target(mask)
            feature = rng.normal(0, 0.01, (1024, 32, 32)).astype(np.float32)
            feature[0] += truth * 3
            semantic = (truth * 0.8 + rng.normal(0, 0.02, truth.shape)).astype(np.float32)
            records.append({"id": str(i), "site": str(i), "mask": mask,
                            "dino": feature, "semantic_raw": semantic, "global_raw": float(positive)})
        model, oof = fit_prior(records, k=1, seed=42)
        self.assertEqual(len(oof), 8)
        self.assertTrue(all(r["id"] in {str(i) for i in range(8)} for r in oof))
        self.assertGreater(model.diagnostics["bank"]["positive_image_coverage"], 0.95)
        inferred = model.predict(records[-1])
        self.assertEqual(inferred["seed_count"], 0)
        self.assertTrue(np.isfinite(inferred["p_spa"]).all())

    def test_synthetic_train_steps(self):
        torch.set_num_threads(2)
        zero = np.zeros((512, 512), dtype=np.float32)
        record = {"id": "synthetic", "site": "synthetic", "mask": zero,
                  "y32": np.zeros((32, 32), dtype=np.float32),
                  "rgb": np.zeros((3, 512, 512), dtype=np.uint8),
                  "dino": np.zeros((1024, 32, 32), dtype=np.float32),
                  "p_sem": np.full((32, 32), 0.1, dtype=np.float32),
                  "p_spa": np.full((32, 32), 0.1, dtype=np.float32),
                  "s_global": 0.1}
        gate, info = train_gate([record], [record], "C6", 0.5, seed=42,
                                epochs=1, lr=1e-4, weight_decay=1e-4, device="cpu")
        self.assertEqual(info["best_epoch"], 1)
        self.assertIsInstance(gate, FusionGate)
        decoder, info = train_decoder([record], [record], "S2", seed=42,
                                      epochs=1, lr=1e-4, weight_decay=1e-4, device="cpu")
        self.assertEqual(info["best_epoch"], 1)
        self.assertIsInstance(decoder, Decoder)


if __name__ == "__main__":
    unittest.main()
