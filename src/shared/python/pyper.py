"""
pyper - Conduit between R and Python for temporal obs-pair climate extraction.

Reads a feather file of observation-pair coordinates, extracts climate window
summaries from geonpy .npy files, and writes results back as feather.

Ported from: OLD_RECA/code/pyper.py
Changes:
  - Replaced deprecated 'feather' package with pyarrow.feather
  - Removed hardcoded CSIRO UNC paths (use -src flag or default to ./data/geonpy)
  - Cleaned up temp file handling
  - Added docstrings

Dependencies:
  - numpy, pandas, pyarrow
  - geonpy (pip install git+https://github.com/cwarecsiro/geonpy.git)
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from geonpy.geonpy import (
    Geonpy,
    calc_climatology_window,
    gen_multi_index_slice,
)

# Default geonpy data source (relative to script location)
DEFAULT_NPY_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "geonpy")


def write_feather(arr, dst, variables):
    """Write output array as feather with named columns."""
    df = pd.DataFrame(arr)
    varnames = [os.path.basename(i)[:-4] for i in variables]
    v1 = ["{}_1".format(i) for i in varnames]
    v2 = ["{}_2".format(i) for i in varnames]
    df.columns = [
        "x_1", "y_1", "year_1", "month_1",
        "x_2", "y_2", "year_2", "month_2",
    ] + v1 + v2
    pf.write_feather(df, dst)


def read_feather(src):
    """Read feather site-pair file and return as numpy array."""
    df = pf.read_feather(src)
    return np.array(df)


def config(args):
    """Parse and validate arguments."""
    pairs = read_feather(vars(args)["filepath"])

    src = vars(args)["variable_source"]
    if not src:
        src = DEFAULT_NPY_SRC

    var = vars(args)["variable_list"]
    variables = ["{}/{}.npy".format(src, i) for i in var]
    check = [os.path.exists(i) for i in variables]
    if not all(check):
        err = [variables[i] for i in range(len(variables)) if not check[i]]
        raise FileNotFoundError("Could not find {}".format(err))

    stat = vars(args)["clim_stat"]
    cstat = getattr(np, stat)

    stat = vars(args)["month_stat"]
    mstat = getattr(np, stat)

    window = int(vars(args)["window"])

    # Temp output file
    dst_dir = vars(args).get("dst", None)
    if dst_dir and os.path.isdir(dst_dir):
        d = tempfile.NamedTemporaryFile(suffix=".feather", dir=dst_dir, delete=False).name
    else:
        d = tempfile.NamedTemporaryFile(suffix=".feather", delete=False).name

    return pairs, variables, mstat, cstat, window, d


def main(args):
    """Main extraction pipeline."""
    pairs, variables, mstat, cstat, window, d = config(args)

    # Output container
    output = np.zeros((pairs.shape[0], (len(variables) * 2))).astype("float32")

    # Sites
    s1_sites = pairs[:, 0:2]
    s2_sites = pairs[:, 4:6]
    sites = [s1_sites, s2_sites]

    # Window = years * months
    window *= 12

    # Temporal indexes
    t1 = pairs[:, 2:4]
    t2 = pairs[:, 6:8]
    s1_slices = gen_multi_index_slice(t1, window, st_year=1911)
    s2_slices = gen_multi_index_slice(t2, window, st_year=1911)
    slices = [s1_slices, s2_slices]

    # Loop over sites and variables
    v_idx = 0
    for s in range(2):
        for v in variables:
            var_v = Geonpy(v)
            arr = var_v.read_points(sites[s], dim_idx=slices[s])
            output[:, v_idx] = calc_climatology_window(arr, mstat, cstat)
            del var_v
            v_idx += 1

    output = pd.DataFrame(np.hstack([pairs, output]))
    write_feather(output, d, variables)

    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract temporal climate windows from geonpy arrays for obs-pairs",
        epilog="Variables available depend on .npy files in -src directory.",
    )
    parser.add_argument(
        "-f", "--filepath", required=True,
        help="Path to feather file with obs-pair coordinates (x1,y1,year1,month1,x2,y2,year2,month2).",
    )
    parser.add_argument(
        "-e", "--variable-list", nargs="+", required=True,
        help="Variable names matching .npy filenames in src (without extension).",
    )
    parser.add_argument(
        "-s", "--clim-stat", required=True,
        help="Numpy stat for climatology summary (e.g. mean, min, max, ptp).",
    )
    parser.add_argument(
        "-m", "--month-stat", required=True,
        help="Numpy stat for monthly summary (e.g. mean, min, max).",
    )
    parser.add_argument(
        "-w", "--window", required=True,
        help="Climate window length in years.",
    )
    parser.add_argument(
        "-d", "--dst",
        help="Destination directory for output feather (uses temp if omitted).",
    )
    parser.add_argument(
        "-src", "--variable-source",
        help="Path to directory containing geonpy .npy files.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print progress messages.",
    )

    args = parser.parse_args()
    print(main(args))
