import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from pvstudy.analysis import L14_EXTENDED, l14_development_report
from pvstudy.experiment import _identity
from pvstudy.io import read_json, sha256, write_json
from pvstudy.review_bundle import export_l14_review
from scripts.run_l14_development import checked_diagnostic


class L14ReportTests(unittest.TestCase):
    def test_missing_or_changed_diagnostic_is_actionable(self):
        with tempfile.TemporaryDirectory() as temporary:
            diagnostic = Path(temporary) / "source" / "diagnostics" / "l14-semantic-v2"
            config = {"semantic_diagnostic_dir": str(diagnostic), "semantic_summary_sha256": "0" * 64}
            with self.assertRaisesRegex(FileNotFoundError, "rebuild_l14_diagnostic.py"):
                checked_diagnostic(config)
            write_json(diagnostic / "l14_summary.json", {"status": "fixture"})
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                checked_diagnostic(config)
            config["semantic_summary_sha256"] = sha256(diagnostic / "l14_summary.json")
            with self.assertRaisesRegex(FileNotFoundError, "0/79 raw maps"):
                checked_diagnostic(config)
            raw = diagnostic / "raw"
            raw.mkdir()
            for number in range(79):
                (raw / f"{number}.npz").touch()
            self.assertEqual(checked_diagnostic(config), diagnostic.parents[1])

    def _make_grid(self, output: Path) -> None:
        manifest = {"images": []}
        split = {"status": "locked_after_manual_qc", "assignment": {}}
        index = {"identity": {"semantic_mode": "window_cls_positive_only",
                              "georsclip_model": "ViT-L-14-336"}, "entries": {}}
        diagnostic = output / "diagnostic"
        write_json(diagnostic / "identity.json", {"protocol": "fixture"})
        write_json(diagnostic / "l14_summary.json", {"status": "fixture"})
        index["identity"]["semantic_source"] = {
            "directory": str(diagnostic),
            "identity_sha256": sha256(diagnostic / "identity.json"),
            "summary_sha256": sha256(diagnostic / "l14_summary.json")}
        for name, payload in (("dataset_manifest.json", manifest), ("split_lock.json", split),
                              ("cache/index.json", index), ("g1_encoder_probe.json", {"status": "fixture"})):
            write_json(output / name, payload)
        write_json(output / "e08_cache_verification.json", {
            "status": "e08_cache_verified",
            "cache_index_sha256": sha256(output / "cache/index.json"),
            "manifest_sha256": sha256(output / "dataset_manifest.json"),
            "split_sha256": sha256(output / "split_lock.json")})
        settings = {"epochs_max": 1}
        identity = _identity(manifest, split, index, settings)
        for fold in range(4):
            for seed in (42, 43, 44):
                def rows(score, fp, metric):
                    return [{"site": f"positive-{fold}", "dice": score,
                             "ap": score, "iou": score,
                             "negative_fp": None},
                            {"site": f"negative-{fold}", "dice": None, "ap": None,
                             "iou": None, "negative_fp": fp}]
                cam = {name: {"site_rows": rows(score, 0.01, "ap")} for name, score in {
                    "C0a": 0.2, "C1": 0.4, "C2": 0.5, "C2m": 0.5,
                    "C4": 0.5, "C5": 0.55, "C5g": 0.6, "C6": 0.62, "C7": 0.5}.items()}
                base = {"fold": fold, "seed": seed, "semantic_mode": "window_cls_positive_only",
                        "cam_identity": identity, "training_config": settings, "cam": cam,
                        "prior": {"bank": {"positive_image_coverage": 1.0}},
                        "bank_drift": {"quality_pass": True}}
                folder = output / f"fold-{fold}" / f"seed-{seed}"
                write_json(folder / "result-cam.json", {**base, "variant": None})
                for variant in L14_EXTENDED:
                    score = {"S2": 0.5, "S14": 0.7, "S6": 0.73, "S7": 0.75}.get(variant, 0.65)
                    fp = {"S2": 0.01, "S14": 0.01, "S6": 0.012, "S7": 0.013}.get(variant, 0.01)
                    write_json(folder / f"result-{variant}.json", {
                        **base, "variant": variant,
                        "segmentation": {"site_rows": rows(score, fp, "dice")}})

    def test_current_l14_summary_and_portable_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self._make_grid(output)
            core = l14_development_report(output)
            self.assertEqual(core["status"], "l14_development_complete")
            self.assertAlmostEqual(core["comparisons"]["S14_vs_S2"]["delta"], 0.2)
            self.assertIsNone(core["comparisons"]["S14_vs_S2"]["quality_pass"])
            self.assertTrue(core["comparisons"]["S6_vs_S14"]["quality_pass"])
            self.assertEqual(core["comparisons"]["S6_vs_S14"]["paired_sites"]["sites"], 4)
            extended = l14_development_report(output, "extended")
            self.assertIn("S6_vs_S11", extended["comparisons"])
            bundle = export_l14_review(output)
            with ZipFile(bundle) as archive:
                names = set(archive.namelist())
                self.assertIn("fold-0/seed-42/result-S14.json", names)
                self.assertIn("diagnostic/l14_summary.json", names)
                self.assertNotIn("fold-0/seed-42/result-S11.json", names)
                metadata = json.loads(archive.read("bundle_manifest.json"))
                self.assertEqual(metadata["files"]["cache/index.json"], sha256(output / "cache/index.json"))
                self.assertEqual(read_json(output / "l14_development_core.json"), core)
            index_path = output / "cache/index.json"
            index = read_json(index_path)
            index["identity"]["tampered"] = True
            write_json(index_path, index)
            with self.assertRaisesRegex(ValueError, "identity differs"):
                l14_development_report(output)


if __name__ == "__main__":
    unittest.main()
