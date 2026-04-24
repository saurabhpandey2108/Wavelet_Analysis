"""
CWT scalogram → image utilities. Pure NumPy + cv2 for resize.

Per-window preprocessing is a cubic polynomial detrend plus z-score
normalization:
  - Cubic detrend removes DC, slow SOC-drift, and low-order curvature.
  - z-score brings every window to unit variance so the CWT magnitudes
    are comparable across windows regardless of voltage/current level.

Scalogram magnitude is expressed in dB (20·log10|CWT|). A COI mask from
cwt_calc/cwt_utils zeroes-out unreliable edge coefficients (set to NaN here;
the pipeline handles NaN in percentile-based normalization stats and then
clamps to 0 after normalization).

Final normalization is global (percentile-based, computed across all
training windows) and applied per-channel (V vs I), then the 2-D image
is resized to 224×224 via bilinear interpolation.
"""

import numpy as np
import cv2

from cwt_calc.cwt_utils import compute_cwt, coi_mask


_EPS_DB = 1e-10   # magnitude floor so log10 stays finite


def preprocess_window(signal):
    """Cubic polynomial detrend + z-score for a 1-D window.

    Removes DC, slow SOC drift, and low-order trend. After this every window
    is zero-mean / unit-variance, so the CWT captures pure dynamic content
    and magnitudes are comparable across windows.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < 4:
        return x - x.mean()
    # Normalize t to [0, 1] for numerical stability of polyfit
    t = np.linspace(0.0, 1.0, n)
    coeffs = np.polyfit(t, x, deg=3)
    trend = np.polyval(coeffs, t)
    x = x - trend
    sd = x.std()
    if sd > 1e-9:
        x = x / sd
    return x


def raw_log_scalogram(signal, fs, scales):
    """dB-magnitude scalogram of a preprocessed window with COI masked out.

    Returns
    -------
    scalogram : (len(scales), len(signal)) float32 ndarray; NaN where COI.
    freqs     : (len(scales),) ndarray of pseudo-frequencies in Hz.
    """
    x = preprocess_window(signal)
    coeffs, freqs = compute_cwt(x, fs, scales)
    mag = np.abs(coeffs)
    db = 20.0 * np.log10(mag + _EPS_DB)

    mask = coi_mask(len(signal), scales)
    db[mask] = np.nan

    return db.astype(np.float32), freqs


def normalize_and_resize(scalogram, vmin, vmax, img_size=(224, 224)):
    """Apply a fixed (vmin, vmax) range, clip to [0, 1], replace any NaN
    (COI-masked pixels) with 0, then bilinear-resize to img_size."""
    span = max(vmax - vmin, 1e-10)
    img = (scalogram - vmin) / span
    img = np.clip(img, 0.0, 1.0)
    img = np.where(np.isnan(img), 0.0, img)   # COI → 0 in the normalized image
    img = img.astype(np.float32)
    return cv2.resize(img, img_size)


def cwt_to_image(signal, fs, img_size=(224, 224), scales=None,
                 vmin=None, vmax=None):
    """Single-window convenience wrapper — same pipeline as training."""
    if scales is None:
        scales = np.geomspace(2.0, 300.0, 96)
    scalogram, freqs = raw_log_scalogram(signal, fs, scales)

    valid = scalogram[~np.isnan(scalogram)]
    if vmin is None:
        vmin = float(np.percentile(valid, 1)) if valid.size else 0.0
    if vmax is None:
        vmax = float(np.percentile(valid, 99)) if valid.size else 1.0

    img = normalize_and_resize(scalogram, vmin, vmax, img_size=img_size)
    return img, freqs


def stack_channels(img_v, img_i):
    """Stack voltage and current images into an HxWx2 array."""
    v = img_v if img_v.ndim == 2 else img_v[..., 0]
    i = img_i if img_i.ndim == 2 else img_i[..., 0]
    return np.stack([v, i], axis=-1)
