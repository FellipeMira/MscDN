#!/usr/bin/env python3
"""
utils/compute_thresholds.py
============================
One-time script to compute the per-pixel P90, P95, P100 from the 30-year
ERA5 NetCDF cube and save them as fixed GeoTIFF artefacts + JSON metadata.

Usage
-----
  python utils/compute_thresholds.py                      # skip if exists
  python utils/compute_thresholds.py --force              # recompute

The resulting rasters are used by the pipeline as the **fixed, immutable**
reference thresholds for all experiments.  They never need to be recomputed
unless the source cube changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.targets import ensure_percentile_rasters


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-pixel P90/P95/P100 from the 30-year NetCDF cube."
    )
    parser.add_argument(
        "--cube",
        type=str,
        default="data/raster/cubo_precipitacao.nc",
        help="Path to the climatological NetCDF cube.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raster",
        help="Directory for the output GeoTIFFs and metadata JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if the rasters already exist.",
    )
    args = parser.parse_args()

    paths = ensure_percentile_rasters(
        cube_nc=args.cube,
        out_dir=args.output_dir,
        force=args.force,
    )
    print(f"\nDone.  Artefacts:")
    for tag, path in paths.items():
        print(f"  {tag:>10s} → {path}")


if __name__ == "__main__":
    main()
