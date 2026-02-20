#!/usr/bin/env python3
"""
utils/compute_p95.py
====================
One-time script to compute the per-pixel P95 from the 30-year ERA5 NetCDF
cube and save it as a fixed GeoTIFF artefact.

Usage
-----
  python utils/compute_p95.py                          # skip if exists
  python utils/compute_p95.py --force                  # recompute

The resulting raster is used by the pipeline as the **fixed, immutable**
P95 reference for all experiments.  It never needs to be recomputed unless
the source cube changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.targets import ensure_p95_raster


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-pixel P95 from the 30-year NetCDF cube."
    )
    parser.add_argument(
        "--cube",
        type=str,
        default="data/raster/cubo_precipitacao.nc",
        help="Path to the climatological NetCDF cube.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raster/p95_precipitacao.tif",
        help="Destination GeoTIFF for the P95 grid.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if the GeoTIFF already exists.",
    )
    args = parser.parse_args()

    result = ensure_p95_raster(
        cube_nc=args.cube,
        p95_tif=args.output,
        force=args.force,
    )
    print(f"\nDone. P95 raster at: {result}")


if __name__ == "__main__":
    main()
