from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
sys.path.insert(0, str(ROOT / "src"))


def _pytest_temp_root_is_usable() -> bool:
    """pytest roots `tmp_path` under a `pytest-of-<user>` directory in the system
    temp area and scans it on startup. On this machine that directory exists and
    cannot be listed even by an administrator shell, which fails every tmp_path
    test before its body runs - a broken scratch directory, not a broken test."""
    import getpass
    import tempfile

    try:
        user = getpass.getuser()
    except Exception:
        user = "unknown"
    root = Path(tempfile.gettempdir()) / f"pytest-of-{user}"
    if not root.exists():
        return True
    try:
        next(os.scandir(root), None)
        return True
    except OSError:
        return False


if not _pytest_temp_root_is_usable():
    # Repo-local and ignored, so the suite stays runnable without touching
    # anything outside the working tree. Only engaged when the default is
    # unusable; a healthy machine keeps pytest's own behaviour.
    import tempfile

    _SCRATCH = ROOT / ".pytest-tmp"
    _SCRATCH.mkdir(exist_ok=True)
    for _var in ("TMPDIR", "TEMP", "TMP"):
        os.environ[_var] = str(_SCRATCH)
    tempfile.tempdir = str(_SCRATCH)
