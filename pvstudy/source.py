from __future__ import annotations

import subprocess
from pathlib import Path


DINOV3_REPOSITORY = "https://github.com/facebookresearch/dinov3.git"


def _git(*arguments: str) -> str:
    process = subprocess.run(["git", *arguments], capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"Git command failed: {process.stderr.strip()}")
    return process.stdout.strip()


def prepare_dinov3_source(path: Path, commit: str) -> dict:
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit.lower()):
        raise ValueError("DINOv3 source commit must be a full 40-character SHA-1")
    path = path.resolve()
    if not path.exists():
        if not path.parent.is_dir():
            raise FileNotFoundError(f"DINOv3 source parent directory is missing: {path.parent}")
        _git("clone", "--filter=blob:none", DINOV3_REPOSITORY, str(path))
        _git("-C", str(path), "checkout", "--detach", commit)
    elif not path.is_dir():
        raise NotADirectoryError(path)
    root = Path(_git("-C", str(path), "rev-parse", "--show-toplevel")).resolve()
    head = _git("-C", str(path), "rev-parse", "HEAD")
    if root != path or head != commit:
        raise ValueError(f"Existing DINOv3 source is not the pinned checkout: root={root}, HEAD={head}")
    if not (path / "dinov3" / "hub" / "backbones.py").is_file():
        raise FileNotFoundError("Pinned DINOv3 checkout lacks dinov3/hub/backbones.py")
    return {"path": str(path), "repository": DINOV3_REPOSITORY, "commit": head}
