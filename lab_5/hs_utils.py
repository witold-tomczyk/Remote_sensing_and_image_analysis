"""Shared helpers for Lab 5 hyperspectral / Sentinel-2 water-quality workflow."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import spectral.io.envi as envi
except ImportError:  # pragma: no cover - optional at import time
    envi = None

# Sentinel-2 MSI central wavelengths (nm) used for spectral resampling.
S2_WAVELENGTHS_NM = {
    "B02": 490.0,
    "B03": 560.0,
    "B04": 665.0,
    "B05": 705.0,
    "B06": 740.0,
    "B07": 783.0,
    "B08": 842.0,
    "B8A": 865.0,
    "B11": 1610.0,
    "B12": 2190.0,
}

# Approximate full-width-half-maximum (nm) for Gaussian band matching.
S2_FWHM_NM = {
    "B02": 65.0,
    "B03": 35.0,
    "B04": 30.0,
    "B05": 15.0,
    "B06": 15.0,
    "B07": 20.0,
    "B08": 115.0,
    "B8A": 20.0,
    "B11": 90.0,
    "B12": 180.0,
}


def require_spectral() -> None:
    if envi is None:
        raise ImportError("Install the spectral package: pip install spectral")


def find_hdr_files(directory: Path) -> list[Path]:
    return sorted(Path(directory).glob("*.hdr"))


def parse_wavelengths(meta: dict) -> np.ndarray | None:
    wl = meta.get("wavelength")
    if wl:
        return np.asarray([float(w) for w in wl], dtype=np.float64)
    return None


def get_rgb_bands(meta: dict, fallback: tuple[int, int, int] = (30, 20, 10)) -> tuple[int, int, int]:
    db = meta.get("default bands")
    if db and len(db) >= 3:
        return tuple(int(float(v)) - 1 for v in db[:3])
    return fallback


def get_ignore_value(meta: dict) -> float | None:
    raw = meta.get("data ignore value")
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def get_reflectance_scale(meta: dict) -> float:
    raw = meta.get("reflectance scale factor")
    if raw is None:
        return 1.0
    try:
        return float(str(raw).strip())
    except ValueError:
        return 1.0


def parse_acquisition_date(meta: dict, hdr_path: Path | None = None) -> str | None:
    """Best-effort acquisition date from ENVI metadata or filename."""
    for key in ("acquisition date", "acquisition time", "date", "flight date"):
        if key in meta and meta[key]:
            raw = str(meta[key]).strip()[:10]
            try:
                datetime.strptime(raw, "%Y-%m-%d")
                return raw
            except ValueError:
                pass

    if hdr_path is not None:
        stem = hdr_path.stem
        prefix = stem.split("_", 1)[0]
        if len(prefix) == 6 and prefix.isdigit():
            yy, mm, dd = int(prefix[:2]), int(prefix[2:4]), int(prefix[4:6])
            year = 2000 + yy if yy < 80 else 1900 + yy
            if dd == 0:
                dd = 1
            return f"{year:04d}-{mm:02d}-{dd:02d}"
    return None


def load_envi(hdr_path: Path):
    require_spectral()
    return envi.open(str(hdr_path))


def mask_invalid(values: np.ndarray, ignore_value: float | None) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    if ignore_value is not None:
        out[out >= ignore_value] = np.nan
    out[out < 0] = np.nan
    return out


def to_reflectance(values: np.ndarray, scale: float) -> np.ndarray:
    if scale and scale != 1.0:
        return values / scale
    return values


def nearest_band_index(wavelengths: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(wavelengths - target_nm)))


def band_at(wavelengths: np.ndarray, cube: np.ndarray, target_nm: float) -> np.ndarray:
    """Return a 2-D band image nearest to target_nm from a 3-D cube (lines, samples, bands)."""
    idx = nearest_band_index(wavelengths, target_nm)
    return cube[:, :, idx]


def percentile_stretch(rgb: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    stretched = rgb.astype(np.float32, copy=True)
    for c in range(stretched.shape[-1]):
        ch = stretched[:, :, c]
        p2, p98 = np.nanpercentile(ch, [p_low, p_high])
        stretched[:, :, c] = np.clip((ch - p2) / max(p98 - p2, 1e-6), 0, 1)
    return np.nan_to_num(stretched, nan=0.0)


def read_rgb_preview(img, wavelengths: np.ndarray | None, meta: dict, ignore_value: float | None) -> np.ndarray:
    r, g, b = get_rgb_bands(meta)
    rgb = img.read_bands([r, g, b]).astype(np.float32)
    rgb = mask_invalid(rgb, ignore_value)
    scale = get_reflectance_scale(meta)
    rgb = to_reflectance(rgb, scale)
    return percentile_stretch(rgb)


def false_color_composite(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    bands_nm: tuple[float, float, float],
    ignore_value: float | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Build a 3-band false-colour composite from target wavelengths (nm)."""
    idx = [nearest_band_index(wavelengths, nm) for nm in bands_nm]
    rgb = cube[:, :, idx].astype(np.float32)
    rgb = mask_invalid(rgb, ignore_value)
    rgb = to_reflectance(rgb, scale)
    return percentile_stretch(rgb)


def compute_ndci(cube: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Normalized Difference Chlorophyll Index (705 vs 665 nm)."""
    red = band_at(wavelengths, cube, 665.0)
    red_edge = band_at(wavelengths, cube, 705.0)
    return (red_edge - red) / np.maximum(red_edge + red, 1e-6)


def compute_doc_proxy(cube: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """DOC proxy: green/red ratio (higher => more dissolved organics)."""
    green = band_at(wavelengths, cube, 560.0)
    red = band_at(wavelengths, cube, 665.0)
    return green / np.maximum(red, 1e-6)


def compute_turbidity_proxy(cube: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Turbidity proxy: NDTI-style (red - green) / (red + green)."""
    green = band_at(wavelengths, cube, 560.0)
    red = band_at(wavelengths, cube, 665.0)
    return (red - green) / np.maximum(red + green, 1e-6)


def align_raster_stack(arrays: dict[str, np.ndarray], reference: str) -> dict[str, np.ndarray]:
    """Upsample bands to the reference band grid (nearest-neighbour, integer factors)."""
    ref = arrays[reference]
    target_shape = ref.shape
    out = {}
    for name, arr in arrays.items():
        if arr.shape == target_shape:
            out[name] = arr
            continue
        sy = target_shape[0] / arr.shape[0]
        sx = target_shape[1] / arr.shape[1]
        if sy != int(sy) or sx != int(sx):
            raise ValueError(f"Cannot align {name}: non-integer scale factors")
        out[name] = np.repeat(np.repeat(arr, int(sy), axis=0), int(sx), axis=1)
    return out


def compute_s2_ndci(b04: np.ndarray, b05: np.ndarray) -> np.ndarray:
    return (b05 - b04) / np.maximum(b05 + b04, 1e-6)


def compute_s2_doc_proxy(b03: np.ndarray, b04: np.ndarray) -> np.ndarray:
    return b03 / np.maximum(b04, 1e-6)


def compute_s2_turbidity_proxy(b03: np.ndarray, b04: np.ndarray) -> np.ndarray:
    return (b04 - b03) / np.maximum(b04 + b03, 1e-6)


def spectral_angle(spectra: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """SAM in radians for spectra shaped (n_pixels, n_bands)."""
    ref = np.asarray(reference, dtype=np.float64)
    ref_norm = np.linalg.norm(ref)
    if ref_norm == 0:
        return np.full(spectra.shape[0], np.nan)
    dots = spectra @ ref
    norms = np.linalg.norm(spectra, axis=1) * ref_norm
    cos_theta = np.clip(dots / np.maximum(norms, 1e-12), -1.0, 1.0)
    return np.arccos(cos_theta)


def convolve_to_s2(spectrum: np.ndarray, wavelengths: np.ndarray) -> dict[str, float]:
    """Resample a 1-D hyperspectral signature to Sentinel-2 bands (Gaussian SRF)."""
    out: dict[str, float] = {}
    for band, center in S2_WAVELENGTHS_NM.items():
        sigma = S2_FWHM_NM[band] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        weights = np.exp(-0.5 * ((wavelengths - center) / max(sigma, 1e-6)) ** 2)
        valid = np.isfinite(spectrum) & (weights > 0)
        if not np.any(valid):
            out[band] = np.nan
            continue
        w = weights[valid]
        out[band] = float(np.average(spectrum[valid], weights=w))
    return out


def convolve_cube_to_s2(cube: np.ndarray, wavelengths: np.ndarray) -> dict[str, np.ndarray]:
    """Convolve full hyperspectral cube to Sentinel-2 band stacks (vectorised)."""
    lines, samples, _ = cube.shape
    flat = cube.reshape(-1, cube.shape[-1])
    result: dict[str, np.ndarray] = {}

    for band, center in S2_WAVELENGTHS_NM.items():
        sigma = S2_FWHM_NM[band] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        weights = np.exp(-0.5 * ((wavelengths - center) / max(sigma, 1e-6)) ** 2)
        weights = weights / np.maximum(weights.sum(), 1e-12)
        conv = np.nansum(flat * weights, axis=1)
        result[band] = conv.reshape(lines, samples)

    return result


def fit_linear_calibration(
    hs_values: np.ndarray,
    s2_values: np.ndarray,
) -> tuple[float, float]:
    """Return slope and intercept: hs ≈ slope * s2 + intercept."""
    mask = np.isfinite(hs_values) & np.isfinite(s2_values)
    if np.count_nonzero(mask) < 10:
        return 1.0, 0.0
    x = s2_values[mask]
    y = hs_values[mask]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def apply_calibration(values: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return slope * values + intercept


def save_spectrum_csv(
    path: Path,
    wavelengths: np.ndarray | None,
    spectrum: np.ndarray,
    class_name: str,
    row: int,
    col: int,
) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_col = "wavelength_nm" if wavelengths is not None else "band"
    x = wavelengths if wavelengths is not None else np.arange(len(spectrum))

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", class_name])
        writer.writerow(["row", row])
        writer.writerow(["col", col])
        writer.writerow([x_col, "value"])
        for xi, vi in zip(x, spectrum):
            writer.writerow([float(xi), "" if np.isnan(vi) else float(vi)])


def load_spectral_library(library_dir: Path) -> dict[str, np.ndarray]:
    """Load spectra saved by build_spectral_library.py."""
    import csv

    library: dict[str, list[np.ndarray]] = {}
    for csv_path in sorted(Path(library_dir).glob("*.csv")):
        with csv_path.open() as f:
            reader = csv.reader(f)
            rows = list(reader)
        class_name = rows[0][1]
        values = []
        data_started = False
        for row in rows:
            if row and row[0] in ("wavelength_nm", "band"):
                data_started = True
                continue
            if data_started and len(row) >= 2 and row[1] != "":
                values.append(float(row[1]))
        if values:
            library.setdefault(class_name, []).append(np.asarray(values, dtype=np.float64))

    return {k: np.vstack(v) for k, v in library.items()}


def bbox_wgs84_from_map_info(meta: dict, ncols: int, nrows: int) -> list[float] | None:
    """Return WGS84 [west, south, east, north] from ENVI map info."""
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    parts = meta.get("map info")
    if not parts or len(parts) < 8:
        return None

    try:
        ulx = float(parts[3])
        uly = float(parts[4])
        px = float(parts[5])
        py = abs(float(parts[6]))
        zone = int(parts[7])
        proj = str(parts[0]).upper()

        lrx = ulx + ncols * px
        lry = uly - nrows * py

        if "UTM" in proj:
            utm_crs = CRS.from_epsg(32600 + zone)
            return list(transform_bounds(utm_crs, CRS.from_epsg(4326), ulx, lry, lrx, uly))
        if "Geographic" in proj or "Lat" in proj:
            return [min(ulx, lrx), min(lry, uly), max(ulx, lrx), max(lry, uly)]
    except (ValueError, TypeError):
        return None

    return None


def save_library_manifest(library_dir: Path, entries: list[dict]) -> None:
    manifest = Path(library_dir) / "manifest.json"
    manifest.write_text(json.dumps(entries, indent=2))
