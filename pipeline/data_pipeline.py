import os
import numpy as np

from data_preprocessing.preprocess import load_signals, estimate_fs, compute_soc, create_windows
from cwt_image.image_utils import raw_log_scalogram, normalize_and_resize, stack_channels
from config.settings import INITIAL_SOC, CAPACITY_AH, WINDOW_SIZE, STRIDE, SCALES
from visualization.plot_utils import plot_voltage_current, plot_soc_profile


def process_dataset_raw(config, results_dir=None):
    """Load one Arbin file (Step 7 only), compute SOC, window, and build raw
    log1p-magnitude scalograms for V and I. No normalization happens here so
    the caller can compute global norm stats across the union of multiple
    datasets before building final images.

    Returns a dict with:
        raw_v, raw_i : (N, n_scales, WINDOW_SIZE) float32
        y            : (N,) SOC labels (at end of each window)
        temperature  : (N,) ambient temperature per window (°C)
        fs           : sampling frequency
        v_win        : (N, WINDOW_SIZE) raw voltage windows — for plotting
        voltage, current, time, soc : full Step-7 signals — for plotting
        label        : human-readable dataset label
    """
    path = config["path"]
    sheet = config["sheet"]
    temp = config["ambient_temp"]
    label = config["label"]

    print(f"\n  ── Loading {label}: {os.path.basename(path)}")
    df, voltage, current, _temperature, time = load_signals(
        path, sheet_name=sheet, ambient_temp=temp
    )
    print(f"    Samples: {len(voltage)} | Temp: {temp}°C")
    print(f"    V range: {voltage.min():.3f} — {voltage.max():.3f} V")
    print(f"    I range: {current.min():.3f} — {current.max():.3f} A")

    fs = estimate_fs(time)
    print(f"    Sampling freq: {fs:.4f} Hz")

    soc = compute_soc(df, initial_soc=INITIAL_SOC, capacity_ah=CAPACITY_AH)
    print(f"    SOC range: {soc.min():.4f} — {soc.max():.4f}")

    if results_dir:
        safe_label = label.replace(" ", "_").replace("°", "").replace("@", "at")
        plot_voltage_current(voltage, current, time,
                             os.path.join(results_dir, f"raw_signals_{safe_label}.png"),
                             title=f"Raw Signals — {label}")
        plot_soc_profile(soc, time,
                         os.path.join(results_dir, f"soc_profile_{safe_label}.png"),
                         title=f"Coulomb Counting SOC — {label}")

    v_win, y_soc = create_windows(voltage, soc, WINDOW_SIZE, STRIDE)
    i_win, _ = create_windows(current, soc, WINDOW_SIZE, STRIDE)
    print(f"    Windows: {len(v_win)} | SOC labels: {y_soc.min():.4f} — {y_soc.max():.4f}")

    print(f"    Computing raw scalograms...")
    raw_v = np.empty((len(v_win), len(SCALES), WINDOW_SIZE), dtype=np.float32)
    raw_i = np.empty_like(raw_v)
    for idx in range(len(v_win)):
        sv, _ = raw_log_scalogram(v_win[idx], fs, SCALES)
        si, _ = raw_log_scalogram(i_win[idx], fs, SCALES)
        raw_v[idx] = sv.astype(np.float32)
        raw_i[idx] = si.astype(np.float32)
        if (idx + 1) % 20 == 0 or (idx + 1) == len(v_win):
            print(f"      {idx + 1}/{len(v_win)} windows processed")

    temperature = np.full(len(v_win), float(temp), dtype=np.float32)
    y = np.asarray(y_soc, dtype=np.float32)

    return {
        "raw_v": raw_v, "raw_i": raw_i,
        "y": y, "temperature": temperature,
        "fs": fs, "v_win": v_win,
        "voltage": voltage, "current": current, "time": time, "soc": soc,
        "label": label,
    }


def compute_norm_stats(raw_v_list, raw_i_list, p_lo=1.0, p_hi=99.0):
    """Percentile-based global normalization stats, ignoring COI-masked (NaN) pixels.

    Uses the [p_lo, p_hi] percentiles across the concatenated training
    scalograms so a single outlier can't crush the [0, 1] range. NaN pixels
    (COI mask from raw_log_scalogram) are excluded before percentile.
    """
    all_v = np.concatenate([r.ravel() for r in raw_v_list])
    all_i = np.concatenate([r.ravel() for r in raw_i_list])
    all_v = all_v[~np.isnan(all_v)]
    all_i = all_i[~np.isnan(all_i)]
    v_min, v_max = np.percentile(all_v, [p_lo, p_hi])
    i_min, i_max = np.percentile(all_i, [p_lo, p_hi])
    return {
        "v_min": float(v_min), "v_max": float(v_max),
        "i_min": float(i_min), "i_max": float(i_max),
    }


def build_images(raw_v, raw_i, norm_stats, img_size=(224, 224)):
    """Apply global (vmin, vmax) normalization and resize every window to 224×224×2."""
    X = np.empty((len(raw_v), img_size[0], img_size[1], 2), dtype=np.float32)
    for idx in range(len(raw_v)):
        img_v = normalize_and_resize(raw_v[idx], norm_stats["v_min"], norm_stats["v_max"],
                                     img_size=img_size)
        img_i = normalize_and_resize(raw_i[idx], norm_stats["i_min"], norm_stats["i_max"],
                                     img_size=img_size)
        X[idx] = stack_channels(img_v, img_i)
    return X
