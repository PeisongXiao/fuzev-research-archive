import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess


ROOT = Path(os.environ["FUZEV_ROOT"]).expanduser().resolve()


def source_revision(directory, package=None):
    """Record an available source revision or installed package version."""
    if directory is not None:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory,
                                           text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    if package is not None:
        try:
            return version(package)
        except PackageNotFoundError:
            pass
    return None


def project_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Project artifact path escapes FUZEV_ROOT")
    return path


def prepare():
    for name in ("weights", "data", "runs", "docs", "reports", "state", "cache"):
        project_path(name).mkdir(parents=True, exist_ok=True)

