# Lab 5 — Hyperspectral data & water quality analysis

Fusion of airborne hyperspectral (ENVI/BSQ) and Sentinel-2 multispectral data for water-quality monitoring.

## Contents

| File | Task | Description |
|---|---|---|
| `viewer.py` | 1 | Desktop BSQ viewer — RGB preview, pixel spectra, CSV export |
| `build_spectral_library.py` | 2 | Extract reference spectra for water, vegetation, forest, soil |
| `water_quality.ipynb` | 3–4 | False-colour composites, water indices, S2 download, SAM classification |
| `hs_utils.py` | — | Shared helpers (indices, SAM, S2 calibration) |
| `generate_test_cube.py` | — | Small synthetic ENVI cube for testing without multi-GB data |
| `TODO.pdf` | — | Full assignment description |

## Setup

```bash
cd lab_5
pip install -r requirements.txt
```

## Data

Download the real Odra hyperspectral cube from OneDrive (link in course materials) and place the `.hdr` + `.bsq` pair in `data/images/`.

For local testing without the full dataset:

```bash
python generate_test_cube.py
```

## Running

### Task 1 — BSQ viewer

```bash
python viewer.py
# or
python viewer.py data/images/221000_Odra_HS_Blok_A_008_VS_join_atm.hdr
```

### Task 2 — Spectral library

Inspect the RGB preview in `viewer.py`, then update pixel coordinates in `build_spectral_library.py` (`DEFAULT_SAMPLES`) if needed:

```bash
python build_spectral_library.py
```

Output: `data/spectral_library/*.csv` + `manifest.json`

### Tasks 3–4 — Notebook

```bash
jupyter lab water_quality.ipynb
```

The notebook will:
1. Build false-colour composites (RGB, CIR, SWIR)
2. Compute Chl-a (NDCI), DOC and turbidity proxies on the airborne cube
3. Download the closest Sentinel-2 L2A scene via Microsoft Planetary Computer
4. Compare indices between sensors
5. Run SAM classification (Lab 3 style) and calibrate Sentinel-2 using airborne match-ups

## Notes

- `.bsq` files are gitignored (7–18 GB each). Only code and the spectral library CSVs are committed.
- The synthetic test cube is also gitignored under `data/images/`.
- Water-quality indices are **proxies** suitable for relative comparison; absolute concentrations require in-situ calibration.
