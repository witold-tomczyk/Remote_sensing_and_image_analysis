#!/usr/bin/env python3
"""Generate a small synthetic ENVI BSQ cube for testing Lab 5 without multi-GB data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent / "data" / "images"
HDR_NAME = "221000_Odra_HS_Blok_A_008_VS_join_atm.hdr"
BSQ_NAME = "221000_Odra_HS_Blok_A_008_VS_join_atm.bsq"

N_SAMPLES = 220
N_LINES = 160
N_BANDS = 456
WAVELENGTHS = np.linspace(400, 1000, N_BANDS)
SCALE = 10000.0
IGNORE = 65535


def _class_spectrum(kind: str) -> np.ndarray:
    wl = WAVELENGTHS
    base = 0.05 + 0.02 * np.sin(wl / 80.0)
    if kind == "water":
        sig = 0.03 + 0.01 * np.exp(-((wl - 560) ** 2) / (2 * 40**2))
        sig[wl < 500] *= 0.4
    elif kind == "green_area":
        sig = 0.08 + 0.12 * np.exp(-((wl - 550) ** 2) / (2 * 35**2))
        sig += 0.06 * np.exp(-((wl - 700) ** 2) / (2 * 25**2))
    elif kind == "forest":
        sig = 0.05 + 0.18 * np.exp(-((wl - 720) ** 2) / (2 * 45**2))
        sig[wl < 500] *= 0.5
    else:  # bare soil
        sig = 0.10 + 0.08 * (wl - 400) / 600
    return (sig + base) * SCALE


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spectra = {
        "water": _class_spectrum("water"),
        "green": _class_spectrum("green_area"),
        "forest": _class_spectrum("forest"),
        "soil": _class_spectrum("bare_soil"),
    }

    cube = np.zeros((N_LINES, N_SAMPLES, N_BANDS), dtype=np.int16)

    # Simple spatial layout mimicking river + riparian vegetation.
    yy, xx = np.mgrid[0:N_LINES, 0:N_SAMPLES]
    water_mask = (xx > 40) & (xx < 110) & (yy > 50) & (yy < 120)
    forest_mask = (xx < 50) | (xx > 150)
    green_mask = (~water_mask) & (~forest_mask) & (yy < 80)
    soil_mask = ~(water_mask | forest_mask | green_mask)

    for mask, spec in [
        (water_mask, spectra["water"]),
        (green_mask, spectra["green"]),
        (forest_mask, spectra["forest"]),
        (soil_mask, spectra["soil"]),
    ]:
        noise = np.random.default_rng(42).normal(0, 150, size=(mask.sum(), N_BANDS))
        cube[mask] = np.clip(spec + noise, 0, IGNORE - 1).astype(np.int16)

    bsq_path = OUT_DIR / BSQ_NAME
    cube.tofile(bsq_path)

    wl_str = ",\n".join(f"{w:.2f}" for w in WAVELENGTHS)
    hdr = f"""ENVI
description = {{Synthetic Odra test cube for Lab 5}}
samples = {N_SAMPLES}
lines = {N_LINES}
bands = {N_BANDS}
header offset = 0
file type = ENVI Standard
data type = 2
interleave = bsq
byte order = 0
data ignore value = {IGNORE}
reflectance scale factor = {SCALE}
default bands = {{120, 80, 40}}
wavelength units = nm
wavelength = {{
{wl_str}
}}
map info = {{UTM, 1, 1, 448915.0, 5800232.0, 2.0, 2.0, 33, North, WGS-84, units=Meters}}
acquisition date = 2022-10-15
"""
    hdr_path = OUT_DIR / HDR_NAME
    hdr_path.write_text(hdr)
    print(f"Wrote {bsq_path} ({bsq_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {hdr_path}")


if __name__ == "__main__":
    main()
