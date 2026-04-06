"""
fix_geometa.py — Re-pickle .geometa files for rasterio >= 1.4 compatibility.

Old rasterio versions serialised CRS as {'data': {'init': 'epsg:4326'}}.
Newer rasterio (>= 1.4) expects a WKT string in __setstate__, so loading
these old pickles raises: ValueError: A string is expected.

This script reads each .geometa with a custom unpickler, converts the CRS
to a modern rasterio.crs.CRS object, and re-writes the pickle in place.

Usage:
    python fix_geometa.py /path/to/geonpy/directory
"""

import glob
import os
import pickle
import shutil
import sys

from rasterio.crs import CRS


class CRSProxy:
    """Stand-in for rasterio.crs.CRS during unpickling of old-format files."""

    def __setstate__(self, state):
        if isinstance(state, dict):
            data = state.get("data", state)
            if "init" in data:
                self._crs = CRS.from_user_input(data["init"])
                return
            if "wkt" in data:
                self._crs = CRS.from_wkt(data["wkt"])
                return
        # Assume WKT string (modern format)
        self._crs = CRS.from_wkt(state)

    def to_crs(self):
        return self._crs


class LegacyCRSUnpickler(pickle.Unpickler):
    """Unpickler that substitutes CRSProxy for rasterio.crs.CRS."""

    def find_class(self, module, name):
        if module == "rasterio.crs" and name == "CRS":
            return CRSProxy
        return super().find_class(module, name)


def fix_geometa(path):
    """Re-pickle a single .geometa file with a modern CRS object."""
    with open(path, "rb") as f:
        meta = LegacyCRSUnpickler(f).load()

    # Replace CRSProxy with real CRS
    if "crs" in meta and isinstance(meta["crs"], CRSProxy):
        meta["crs"] = meta["crs"].to_crs()

    # Write back
    with open(path, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <geonpy_directory>")
        sys.exit(1)

    geonpy_dir = sys.argv[1]
    files = sorted(glob.glob(os.path.join(geonpy_dir, "*.geometa")))

    if not files:
        print(f"No .geometa files found in {geonpy_dir}")
        sys.exit(1)

    print(f"Found {len(files)} .geometa files in {geonpy_dir}")

    fixed = 0
    skipped = 0
    for path in files:
        basename = os.path.basename(path)
        try:
            # First check if it already loads fine with standard pickle
            with open(path, "rb") as f:
                pickle.load(f)
            print(f"  OK (already compatible): {basename}")
            skipped += 1
        except (ValueError, TypeError):
            # Needs fixing
            backup = path + ".bak"
            shutil.copy2(path, backup)
            fix_geometa(path)
            # Verify
            with open(path, "rb") as f:
                meta = pickle.load(f)
            print(f"  FIXED: {basename}  (backup: {os.path.basename(backup)})")
            fixed += 1

    print(f"\nDone. Fixed: {fixed}, Already OK: {skipped}, Total: {len(files)}")


if __name__ == "__main__":
    main()
