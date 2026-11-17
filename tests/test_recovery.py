import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from pvstudy.recovery import RECOVERY_FILES, RECOVERY_INFO, RECOVERY_SCHEMA, restore_e05, reuse_locked_split


class RecoveryTests(unittest.TestCase):
    def _payload(self):
        assignment = {str(i): {"site": str(i), "partition": "dev", "fold": i} for i in range(4)}
        manifest = {"manual_qc_confirmed": True, "manual_qc_sha256": "qc",
                    "images": [{"id": i, "image_sha256": "image"} for i in assignment]}
        split = {"status": "locked_after_manual_qc", "qc_sha256": "qc", "assignment": assignment}
        index = {"entries": {i: {"image_sha256": "image"} for i in assignment}}
        objects = dict(zip(RECOVERY_FILES[:3], (manifest, split, index)))
        for fold in range(4):
            objects[f"fold-{fold}/seed-42/result-cam.json"] = {
                "fold": fold, "seed": 42, "variant": None,
                "train_ids": sorted(i for i in assignment if i != str(fold)),
                "validation_ids": [str(fold)], "cam": {"C1": {"site_rows": [{"site": str(fold), "ap": 0.2}]}}}
        files = {n: json.dumps(v).encode() for n,v in objects.items()}
        metadata = {"schema": RECOVERY_SCHEMA, "files": {n: hashlib.sha256(v).hexdigest() for n,v in files.items()}}
        return files, metadata

    def test_zip_directory_resume_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/"input"
            source.mkdir()
            files, metadata = self._payload()
            with ZipFile(source/"pvstudy-e05-recovery.zip", "w") as archive:
                archive.writestr(RECOVERY_INFO, json.dumps(metadata))
                for name, data in files.items():
                    archive.writestr(name, data)
            output = root/"output"
            first = restore_e05(source, output)
            self.assertFalse(first["feature_arrays_restored"])
            for name,data in files.items():
                self.assertEqual((output/name).read_bytes(), data)
            self.assertEqual(first, restore_e05(source, output))
            (source/RECOVERY_INFO).write_text(json.dumps(metadata), encoding="utf-8")
            for name,data in files.items():
                path = source/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            self.assertEqual(restore_e05(source, output)["files"], 7)
            (output/"dataset_manifest.json").write_text("preserve me", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "preserved without overwrite"):
                restore_e05(source, output)
            self.assertEqual((output/"dataset_manifest.json").read_text(), "preserve me")

    def test_corruption_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/"input"
            source.mkdir()
            files, metadata = self._payload()
            (source/RECOVERY_INFO).write_text(json.dumps(metadata), encoding="utf-8")
            for name,data in files.items():
                path = source/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            (source/"split_lock.json").write_text("corrupt", encoding="utf-8")
            output = root/"output"
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                restore_e05(source, output)
            self.assertFalse(output.exists())

    def test_reuse_lock_preserves_exact_bytes_and_refuses_different_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "old"
            source.mkdir()
            files, _ = self._payload()
            for name in RECOVERY_FILES[:2]:
                (source / name).write_bytes(files[name])
            target = root / "new"
            self.assertEqual(reuse_locked_split(source, target)["status"], "locked_split_reused")
            for name in RECOVERY_FILES[:2]:
                self.assertEqual((target / name).read_bytes(), files[name])
            self.assertEqual(reuse_locked_split(source, target)["status"], "locked_split_reused")
            (target / "split_lock.json").write_text("different", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "preserved"):
                reuse_locked_split(source, target)


if __name__ == "__main__":
    unittest.main()
