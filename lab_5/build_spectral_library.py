#!/usr/bin/env python3
"""
Task 2 — build a spectral library from an ENVI hyperspectral cube.

Pick representative pixels for water, vegetation, forest, bare soil, etc.
and export their spectra to data/spectral_library/.

Usage:
    python build_spectral_library.py
    python build_spectral_library.py data/images/your_scene.hdr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hs_utils import (
    find_hdr_files,
    get_ignore_value,
    get_reflectance_scale,
    load_envi,
    mask_invalid,
    parse_acquisition_date,
    parse_wavelengths,
    save_spectrum_csv,
    to_reflectance,
)

DATA_DIR = Path(__file__).parent / "data" / "images"
LIBRARY_DIR = Path(__file__).parent / "data" / "spectral_library"

# Default pixel coordinates (row, col) for the Odra test / real scenes.
# Update these after inspecting the RGB preview in viewer.py.
DEFAULT_SAMPLES = {
    "water": [(80, 60), (95, 72)],
    "green_area": [(40, 120), (55, 130)],
    "forest": [(30, 40), (35, 55)],
    "bare_soil": [(110, 140), (120, 155)],
}


def build_library(hdr_path: Path, samples: dict[str, list[tuple[int, int]]], out_dir: Path) -> list[dict]:
    img = load_envi(hdr_path)
    meta = img.metadata
    wavelengths = parse_wavelengths(meta)
    ignore = get_ignore_value(meta)
    scale = get_reflectance_scale(meta)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for class_name, pixels in samples.items():
        for row, col in pixels:
            if not (0 <= row < img.nrows and 0 <= col < img.ncols):
                print(f"Skipping {class_name} ({row}, {col}) — out of bounds")
                continue

            spec = mask_invalid(img.read_pixel(row, col), ignore)
            spec = to_reflectance(spec, scale)

            fname = f"{class_name}_r{row}_c{col}.csv"
            out_path = out_dir / fname
            save_spectrum_csv(out_path, wavelengths, spec, class_name, row, col)

            entry = {
                "class": class_name,
                "row": row,
                "col": col,
                "file": fname,
                "hdr": hdr_path.name,
            }
            manifest.append(entry)
            print(f"Saved {class_name:12s} pixel ({row:4d}, {col:4d}) -> {out_path.name}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_hdr": str(hdr_path),
                "acquisition_date": parse_acquisition_date(meta, hdr_path),
                "entries": manifest,
            },
            indent=2,
        )
    )
    print(f"\nManifest written to {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hyperspectral spectral library (Lab 5, Task 2)")
    parser.add_argument("hdr", nargs="?", help="Path to ENVI .hdr file")
    parser.add_argument(
        "--samples-json",
        type=Path,
        help="JSON file mapping class names to [[row, col], ...] pixel lists",
    )
    parser.add_argument("--out", type=Path, default=LIBRARY_DIR, help="Output directory")
    args = parser.parse_args()

    if args.hdr:
        hdr_path = Path(args.hdr)
    else:
        hdrs = find_hdr_files(DATA_DIR)
        if not hdrs:
            raise SystemExit(f"No .hdr files found in {DATA_DIR}. Place data there or pass a path.")
        hdr_path = hdrs[0] if len(hdrs) == 1 else hdrs[0]
        if len(hdrs) > 1:
            print(f"Multiple HDR files found; using {hdr_path.name}")

    if args.samples_json:
        samples = {k: [tuple(p) for p in v] for k, v in json.loads(args.samples_json.read_text()).items()}
    else:
        samples = DEFAULT_SAMPLES

    build_library(hdr_path, samples, args.out)


if __name__ == "__main__":
    main()
