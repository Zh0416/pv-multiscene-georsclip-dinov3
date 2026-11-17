import tempfile
import unittest
from pathlib import Path

from pvstudy.analysis import cam_development_gates
from pvstudy.experiment import _identity
from pvstudy.io import read_json, write_json


class CamAnalysisTests(unittest.TestCase):
    def test_cam_gates_require_complete_consistent_three_seed_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            incomplete = cam_development_gates(output)
            self.assertEqual(incomplete["status"], "incomplete")
            self.assertEqual(len(incomplete["missing_result_files"]), 12)
            manifest, split, index = {"images": []}, {"status": "locked_after_manual_qc", "assignment": {}}, {
                "identity": {"semantic_mode": "window_cls_positive_only"}, "entries": {}}
            write_json(output / "dataset_manifest.json", manifest)
            write_json(output / "split_lock.json", split)
            write_json(output / "cache" / "index.json", index)
            settings = {"epochs_max": 1}
            identity = _identity(manifest, split, index, settings)
            for fold in range(4):
                for seed in (42, 43, 44):
                    def rows(ap, fp):
                        return [{"site": f"positive-{fold}", "ap": ap, "negative_fp": None},
                                {"site": f"negative-{fold}", "ap": None, "negative_fp": fp}]

                    cam = {name: {"site_rows": rows(ap, 0.01)} for name, ap in {
                        "C0a": 0.2, "C1": 0.4, "C2": 0.35, "C2m": 0.3,
                        "C4": 0.3, "C5": 0.32, "C5g": 0.32, "C6": 0.45}.items()}
                    write_json(output / f"fold-{fold}" / f"seed-{seed}" / "result-cam.json", {
                        "fold": fold, "seed": seed, "variant": None, "semantic_mode": "window_cls_positive_only",
                        "cam_identity": identity, "training_config": settings, "cam": cam,
                        "prior": {"bank": {"positive_image_coverage": 1.0}},
                        "bank_drift": {"quality_pass": True}})
            result = cam_development_gates(output)
            self.assertEqual(result["status"], "development_cam_grid_complete")
            self.assertTrue(result["G2"]["quality_pass"])
            self.assertTrue(result["G3"]["quality_pass"])
            self.assertEqual(result["semantic_mode"], "window_cls_positive_only")
            target = output / "fold-0" / "seed-42" / "result-cam.json"
            changed = read_json(target)
            changed["cam_identity"] = "other-cache"
            write_json(target, changed)
            with self.assertRaisesRegex(ValueError, "mixed cache"):
                cam_development_gates(output)
            changed["cam_identity"] = identity
            write_json(target, changed)
            index["identity"]["unexpected_change"] = True
            write_json(output / "cache" / "index.json", index)
            with self.assertRaisesRegex(ValueError, "differs from current locked cache"):
                cam_development_gates(output)


if __name__ == "__main__":
    unittest.main()
