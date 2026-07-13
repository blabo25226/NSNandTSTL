"""Colab bootstrap: locate TSTL/src and add to sys.path (import this before llm_*)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/blabo25226/NSNandTSTL.git"
# Branch that carries the R1 pipeline code (update to the default branch after merge).
BRANCH = "claude/tstl-reproduction-coding-s267ic"
CLONE_DIR = Path("/content/NSNandTSTL")
DRIVE_REPO = Path("/content/drive/MyDrive/NSNandTSTL")


def _has_src(root: Path) -> bool:
    return (root / "src" / "llm_freeze.py").is_file()


def _git(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _clone() -> Path:
    if CLONE_DIR.exists():
        import shutil

        shutil.rmtree(CLONE_DIR)
    _git(
        ["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, str(CLONE_DIR)],
        cwd=Path("/content"),
    )
    return CLONE_DIR / "TSTL"


def _update_clone() -> Path:
    import shutil

    try:
        _git(["git", "fetch", "origin", BRANCH], cwd=CLONE_DIR)
        _git(["git", "checkout", BRANCH], cwd=CLONE_DIR)
        _git(["git", "reset", "--hard", f"origin/{BRANCH}"], cwd=CLONE_DIR)
    except subprocess.CalledProcessError as exc:
        print(f"git sync failed ({exc}); re-cloning")
        shutil.rmtree(CLONE_DIR)
        return _clone()
    return CLONE_DIR / "TSTL"


def ensure_tstl_root() -> Path:
    """Return TSTL project root; clone or update if needed."""
    if _has_src(DRIVE_REPO / "TSTL"):
        root = DRIVE_REPO / "TSTL"
        print(f"Using Drive: {root}")
    elif CLONE_DIR.is_dir() and (CLONE_DIR / ".git").is_dir():
        root = _update_clone()
        print(f"Updated clone: {root}")
        if not _has_src(root):
            root = _clone()
            print(f"Re-cloned (src was missing): {root}")
    elif _has_src(CLONE_DIR / "TSTL"):
        root = CLONE_DIR / "TSTL"
        print(f"Using clone: {root}")
    else:
        root = _clone()
        print(f"Cloned: {root}")

    if not _has_src(root):
        raise FileNotFoundError(
            f"llm_freeze.py not found under {root / 'src'}. "
            f"Push branch {BRANCH} to GitHub or copy repo to Drive."
        )
    return root


def ensure_src_on_path(root: Path | None = None) -> Path:
    """Add TSTL/src to sys.path and chdir to project root."""
    root = root or ensure_tstl_root()
    src = root / "src"
    src_str = str(src.resolve())
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    os.chdir(root)
    return src


def install_requirements(root: Path) -> None:
    req = root / "requirements-r.txt"
    if not req.is_file():
        raise FileNotFoundError(req)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
        check=True,
    )
