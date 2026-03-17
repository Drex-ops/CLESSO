#!/usr/bin/env python3
"""
backup_snapshot.py -- Save a date-stamped snapshot of the clesso_nn source code.

Creates a copy of all Python, R, and TeX source files (excluding transient
artefacts like __pycache__, output/, backup/, and LaTeX build files) into:

    src/clesso_nn/backup/{YYYY-MM-DD}/

If the target directory already exists a numeric suffix is appended
(e.g. 2026-03-17_2) to avoid overwriting earlier snapshots from the same day.

Usage:
    python src/clesso_nn/backup_snapshot.py            # uses today's date
    python src/clesso_nn/backup_snapshot.py 2026-03-15  # explicit date
"""

from __future__ import annotations

import shutil
import sys
from datetime import date
from pathlib import Path

# Files / dirs to skip when copying
EXCLUDE_DIRS = {"__pycache__", "output", "backup", ".DS_Store"}
EXCLUDE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd",        # compiled Python
    ".aux", ".log", ".out",        # LaTeX build artefacts
    ".gz", ".pdf",                 # LaTeX output (.synctex.gz, .pdf)
    ".DS_Store",
}


def _should_skip(path: Path) -> bool:
    """Return True if *path* should be excluded from the snapshot."""
    if path.name in EXCLUDE_DIRS:
        return True
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return True
    if path.suffix in EXCLUDE_EXTENSIONS:
        return True
    return False


def backup_snapshot(label: str | None = None) -> Path:
    """Copy clesso_nn source into backup/{label} and return the dest path."""
    src_dir = Path(__file__).resolve().parent  # src/clesso_nn/

    if label is None:
        label = date.today().isoformat()  # e.g. "2026-03-17"

    backup_root = src_dir / "backup"
    dest = backup_root / label

    # Avoid overwriting an existing snapshot from the same day
    if dest.exists():
        n = 2
        while (backup_root / f"{label}_{n}").exists():
            n += 1
        dest = backup_root / f"{label}_{n}"

    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    for item in sorted(src_dir.rglob("*")):
        if _should_skip(item):
            continue
        rel = item.relative_to(src_dir)
        # Skip anything already inside the backup tree
        if rel.parts[0] == "backup":
            continue
        target = dest / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            copied += 1

    print(f"Snapshot saved: {dest}  ({copied} files)")
    return dest


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else None
    backup_snapshot(label)
