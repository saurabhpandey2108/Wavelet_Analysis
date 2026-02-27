import numpy as np
import cv2
from cwt_calc.cwt_utils import compute_cwt


def cwt_to_image(signal, fs, img_size=(224, 224), scales=None,
                 temperature_value=None, temp_range=None):
    """Convert a 1D signal to a normalized scalogram image.

    If `temperature_value` is provided, an extra channel filled with the
    normalized temperature is appended (so the result will have shape HxWx2).

    temp_range: optional (min, max) to normalize temperature; if not provided
    the function uses min=temperature_value-5, max=temperature_value+5 as a safe fallback.
    """
    if scales is None:
        scales = np.arange(1, 128)
        
    coeffs, freqs = compute_cwt(signal, fs, scales)
    scalogram = np.abs(coeffs)

    # Normalize to [0,1]
    if scalogram.max() - scalogram.min() == 0:
        img = np.zeros_like(scalogram)
    else:
        img = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min())

    # cv2.resize expects (width, height)
    img_resized = cv2.resize(img, img_size)

    if temperature_value is None:
        return img_resized, freqs

    # Normalize temperature_value
    if temp_range is None:
        tmin = temperature_value - 5.0
        tmax = temperature_value + 5.0
    else:
        tmin, tmax = temp_range

    if tmax - tmin == 0:
        t_norm = 0.0
    else:
        t_norm = (temperature_value - tmin) / (tmax - tmin)
        t_norm = float(np.clip(t_norm, 0.0, 1.0))

    temp_img = np.full_like(img_resized, fill_value=t_norm, dtype=np.float32)
    combined = np.stack([img_resized, temp_img], axis=-1)
    return combined, freqs


def stack_channels(img_v, img_i, temperature_value=None, temp_range=None):
    """Stack voltage and current images and optionally append temperature as third channel.

    img_v, img_i: HxW arrays
    returns HxWx2 or HxWx3
    """
    v = img_v if img_v.ndim == 2 else img_v[..., 0]
    i = img_i if img_i.ndim == 2 else img_i[..., 0]
    base = np.stack([v, i], axis=-1)
    if temperature_value is None:
        return base
    # append temp channel
    if temp_range is None:
        tmin = temperature_value - 5.0
        tmax = temperature_value + 5.0
    else:
        tmin, tmax = temp_range
    if tmax - tmin == 0:
        t_norm = 0.0
    else:
        t_norm = (temperature_value - tmin) / (tmax - tmin)
        t_norm = float(np.clip(t_norm, 0.0, 1.0))
    temp_img = np.full((base.shape[0], base.shape[1], 1), fill_value=t_norm, dtype=np.float32)
    return np.concatenate([base, temp_img], axis=-1)
