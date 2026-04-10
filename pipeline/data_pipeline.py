import os
import numpy as np

from data_preprocessing.preprocess import load_signals, estimate_fs, compute_soc, create_windows
from cwt_image.image_utils import cwt_to_image, stack_channels
from config.settings import INITIAL_SOC, CAPACITY_AH, WINDOW_SIZE, STRIDE
from visualization.plot_utils import plot_voltage_current, plot_soc_profile

def process_dataset(config, results_dir=None):
    """Load one XLS file, compute SOC, create windows, generate CWT images.

    Returns X (images), Y (SOC labels), fs (sampling freq), and intermediate data.
    """
    path = config["path"]
    sheet = config["sheet"]
    temp = config["ambient_temp"]
    label = config["label"]

    print(f"\n  ── Loading {label}: {os.path.basename(path)}")
    df, voltage, current, temperature, time = load_signals(
        path, sheet_name=sheet, ambient_temp=temp
    )
    print(f"    Samples: {len(voltage)} | Temp: {temp}°C")
    print(f"    V range: {voltage.min():.3f} — {voltage.max():.3f} V")
    print(f"    I range: {current.min():.3f} — {current.max():.3f} A")

    fs = estimate_fs(time)
    print(f"    Sampling freq: {fs:.4f} Hz")

    # Compute SOC via Coulomb Counting
    soc = compute_soc(current, time, initial_soc=INITIAL_SOC, capacity_ah=CAPACITY_AH)
    print(f"    SOC range: {soc.min():.4f} — {soc.max():.4f}")

    # Optional: save signal plots
    if results_dir:
        safe_label = label.replace(" ", "_").replace("°", "").replace("@", "at")
        plot_voltage_current(voltage, current, time,
                             os.path.join(results_dir, f"raw_signals_{safe_label}.png"),
                             title=f"Raw Signals — {label}")
        plot_soc_profile(soc, time,
                         os.path.join(results_dir, f"soc_profile_{safe_label}.png"),
                         title=f"Coulomb Counting SOC — {label}")

    # Create sliding windows
    v_win, y_soc = create_windows(voltage, soc, WINDOW_SIZE, STRIDE)
    i_win, _ = create_windows(current, soc, WINDOW_SIZE, STRIDE)
    print(f"    Windows: {len(v_win)} | SOC labels: {y_soc.min():.4f} — {y_soc.max():.4f}")

    # Generate CWT scalogram images
    print(f"    Generating CWT images...")
    X_images = []
    for idx in range(len(v_win)):
        img_v, _ = cwt_to_image(v_win[idx], fs)
        img_i, _ = cwt_to_image(i_win[idx], fs)
        cwt_combined = stack_channels(img_v, img_i)
        X_images.append(cwt_combined)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(v_win):
            print(f"      {idx + 1}/{len(v_win)} windows processed")

    X = np.array(X_images, dtype=np.float32)
    Y = np.array(y_soc, dtype=np.float32)

    return X, Y, fs, v_win, voltage, current, time, soc
