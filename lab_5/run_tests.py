#!/usr/bin/env python3
"""Smoke test for Lab 5 without Jupyter (avoids kernel OOM during CI)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from hs_utils import (
    compute_doc_proxy,
    compute_ndci,
    compute_s2_ndci,
    compute_turbidity_proxy,
    convolve_cube_to_s2,
    convolve_to_s2,
    fit_linear_calibration,
    get_ignore_value,
    get_reflectance_scale,
    load_envi,
    load_spectral_library,
    mask_invalid,
    parse_wavelengths,
    spectral_angle,
    to_reflectance,
)

LAB = Path(__file__).parent
DATA = LAB / "data" / "images"
HDR = DATA / "221000_Odra_HS_Blok_A_008_VS_join_atm.hdr"


def main() -> None:
    if not HDR.exists():
        subprocess.run([sys.executable, "generate_test_cube.py"], check=True, cwd=LAB)

    subprocess.run([sys.executable, "build_spectral_library.py", str(HDR)], check=True, cwd=LAB)

    img = load_envi(HDR)
    wavelengths = parse_wavelengths(img.metadata)
    cube = to_reflectance(
        mask_invalid(np.asarray(img.load()).astype(np.float64), get_ignore_value(img.metadata)),
        get_reflectance_scale(img.metadata),
    )

    assert compute_ndci(cube, wavelengths).shape == cube.shape[:2]
    assert compute_doc_proxy(cube, wavelengths).shape == cube.shape[:2]
    assert compute_turbidity_proxy(cube, wavelengths).shape == cube.shape[:2]

    library = load_spectral_library(LAB / "data" / "spectral_library")
    assert len(library) >= 4

    ref = {cls: np.nanmedian(specs, axis=0) for cls, specs in library.items()}
    flat = cube.reshape(-1, cube.shape[-1])
    for cls, r in ref.items():
        sam = spectral_angle(flat, r)
        assert sam.shape[0] == flat.shape[0]

    hs_s2 = convolve_cube_to_s2(cube, wavelengths)
    hs_ndci = compute_s2_ndci(hs_s2["B04"], hs_s2["B05"]).ravel()
    fake_s2 = hs_ndci + np.random.default_rng(0).normal(0, 0.01, hs_ndci.size)
    slope, intercept = fit_linear_calibration(hs_ndci, fake_s2)
    assert np.isfinite(slope)

    for cls, r in ref.items():
        conv = convolve_to_s2(r, wavelengths)
        assert "B04" in conv

    print("All Lab 5 smoke tests passed.")


if __name__ == "__main__":
    main()
