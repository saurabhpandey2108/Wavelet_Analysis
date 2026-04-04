"""
CWT-CNN Battery SOC Estimation Pipeline
========================================

End-to-end pipeline for estimating the State of Charge (SOC) of Li-ion batteries
using Continuous Wavelet Transform scalograms and a Convolutional Neural Network.

Dataset: Arbin DST profile at 25°C ambient, starting at 50% SOC
         (11_05_2015_SP20-2_DST_50SOC.xls)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.model_selection import train_test_split

from data_preprocessing.preprocess import load_signals, estimate_fs, compute_soc, create_windows
from cwt_image.image_utils import cwt_to_image, stack_channels
from cwt_calc.cwt_utils import compute_cwt
from cnn_model.model import build_cnn_model

# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATASET_FILE = "11_05_2015_SP20-2_DST_50SOC.xls"
RESULTS_DIR = "results"
INITIAL_SOC = 0.5          # Dataset starts at 50% SOC
CAPACITY_AH = 2.0          # SP20 rated capacity (Ah)
WINDOW_SIZE = 256           # Samples per window (~256s at ~1 Hz)
STRIDE = 128                # 50% overlap
SCALES = np.arange(1, 128)  # CWT scales
EPOCHS = 30
BATCH_SIZE = 32
TEST_SPLIT = 0.2
RANDOM_SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
#  Visualization Functions
# ──────────────────────────────────────────────────────────────────────────────

def plot_frequency_scalogram(v_win, fs, freqs, save_path):
    """2D scalogram plot with pseudo-frequency y-axis."""
    coeffs, _ = compute_cwt(v_win, fs, SCALES)
    scalogram = np.abs(coeffs)
    scalogram = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min() + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(scalogram, aspect="auto", cmap="jet", origin="lower",
                   extent=[0, len(v_win) / fs, freqs.min(), freqs.max()])
    fig.colorbar(im, ax=ax, label="Normalized Magnitude")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Pseudo-frequency (Hz)", fontsize=12)
    ax.set_title("CWT Scalogram — Voltage Signal (DST Profile)", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved 2D scalogram → {save_path}")


def plot_3d_scalogram(v_win, fs, freqs, save_path):
    """3D surface plot of the CWT scalogram.

    X-axis: Time (s)
    Y-axis: Pseudo-frequency (Hz)
    Z-axis: CWT Magnitude (normalized)

    This visualization reveals the time-frequency energy distribution
    in three dimensions, making peaks and ridges more visible than
    the standard 2D heatmap.
    """
    coeffs, _ = compute_cwt(v_win, fs, SCALES)
    scalogram = np.abs(coeffs)
    scalogram = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min() + 1e-10)

    time_axis = np.linspace(0, len(v_win) / fs, scalogram.shape[1])
    T, F = np.meshgrid(time_axis, freqs)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(T, F, scalogram, cmap='jet', edgecolor='none',
                           alpha=0.9, rstride=2, cstride=2)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Normalized Magnitude")

    ax.set_xlabel("Time (s)", fontsize=11, labelpad=10)
    ax.set_ylabel("Pseudo-frequency (Hz)", fontsize=11, labelpad=10)
    ax.set_zlabel("CWT Magnitude", fontsize=11, labelpad=10)
    ax.set_title("3D CWT Scalogram — Voltage Signal (DST Profile)", fontsize=14, pad=20)
    ax.view_init(elev=30, azim=225)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved 3D scalogram → {save_path}")


def plot_soc_profile(soc, time, save_path):
    """Plot the computed SOC profile over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time / 3600, soc, color='#2196F3', linewidth=1.5)
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("SOC", fontsize=12)
    ax.set_title("Coulomb Counting SOC Profile — DST at 25°C, Initial SOC = 50%", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved SOC profile → {save_path}")


def plot_voltage_current(voltage, current, time, save_path):
    """Plot raw voltage and current signals."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(time / 3600, voltage, color='#FF5722', linewidth=0.8)
    ax1.set_ylabel("Voltage (V)", fontsize=12)
    ax1.set_title("DST Profile — Raw Signals (25°C Ambient)", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(time / 3600, current, color='#4CAF50', linewidth=0.8)
    ax2.set_xlabel("Time (hours)", fontsize=12)
    ax2.set_ylabel("Current (A)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved V/I plot → {save_path}")


def plot_training_history(history, save_path):
    """Plot training and validation loss/metrics curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [('loss', 'MSE Loss'), ('mae', 'MAE'), ('rmse', 'RMSE'), ('r_squared', 'R²')]

    for ax, (key, label) in zip(axes.flat, metrics):
        if key in history.history:
            ax.plot(history.history[key], label=f'Train {label}', linewidth=1.5)
            val_key = f'val_{key}'
            if val_key in history.history:
                ax.plot(history.history[val_key], label=f'Val {label}',
                        linewidth=1.5, linestyle='--')
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(label, fontsize=13)
            ax.legend()
            ax.grid(True, alpha=0.3)

    fig.suptitle("Training History — CWT-CNN SOC Estimation", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved training history → {save_path}")


def plot_predictions(y_true, y_pred, save_path):
    """Scatter plot of predicted vs actual SOC + error distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter: Predicted vs Actual
    ax1.scatter(y_true, y_pred, alpha=0.5, s=20, c='#1976D2', edgecolors='none')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Ideal (y = x)')
    ax1.set_xlabel("Actual SOC", fontsize=12)
    ax1.set_ylabel("Predicted SOC", fontsize=12)
    ax1.set_title("Predicted vs Actual SOC", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')

    # Error distribution
    errors = y_pred.flatten() - y_true
    ax2.hist(errors, bins=30, color='#FF7043', edgecolor='white', alpha=0.85)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel("Prediction Error (Predicted − Actual)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title(f"Error Distribution (μ={errors.mean():.4f}, σ={errors.std():.4f})", fontsize=14)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved predictions plot → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  CWT-CNN Battery SOC Estimation Pipeline")
    print("  Dataset: DST Profile | 25°C | Initial SOC = 50%")
    print("=" * 70)
    np.random.seed(RANDOM_SEED)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load Data ─────────────────────────────────────────────────────
    dataset_path = os.path.join("dataset", DATASET_FILE)
    print(f"\n[1/7] Loading dataset: {dataset_path}")
    df, voltage, current, temperature, time = load_signals(dataset_path)
    print(f"  Loaded {len(voltage)} data points")
    print(f"  Voltage range: {voltage.min():.3f} V — {voltage.max():.3f} V")
    print(f"  Current range: {current.min():.3f} A — {current.max():.3f} A")
    print(f"  Ambient Temperature: {temperature[0]:.1f}°C (fixed)")

    fs = estimate_fs(time)
    print(f"  Sampling Frequency: {fs:.4f} Hz (dt ≈ {1/fs:.2f} s)")

    # ── 2. Plot Raw Signals ──────────────────────────────────────────────
    print(f"\n[2/7] Plotting raw signals...")
    plot_voltage_current(voltage, current, time,
                         os.path.join(RESULTS_DIR, "raw_signals.png"))

    # ── 3. Compute SOC via Coulomb Counting ──────────────────────────────
    print(f"\n[3/7] Computing SOC (initial={INITIAL_SOC}, capacity={CAPACITY_AH} Ah)...")
    soc = compute_soc(current, time, initial_soc=INITIAL_SOC, capacity_ah=CAPACITY_AH)
    print(f"  SOC range: {soc.min():.4f} — {soc.max():.4f}")
    plot_soc_profile(soc, time, os.path.join(RESULTS_DIR, "soc_profile.png"))

    # ── 4. Create Sliding Windows ────────────────────────────────────────
    print(f"\n[4/7] Creating sliding windows (size={WINDOW_SIZE}, stride={STRIDE})...")
    v_win, y_soc = create_windows(voltage, soc, WINDOW_SIZE, STRIDE)
    i_win, _ = create_windows(current, soc, WINDOW_SIZE, STRIDE)
    print(f"  Total windows: {len(v_win)}")
    print(f"  SOC label range: {y_soc.min():.4f} — {y_soc.max():.4f}")

    # ── 5. Generate Scalogram Visualizations ─────────────────────────────
    print(f"\n[5/7] Generating scalogram visualizations...")
    _, freqs = cwt_to_image(v_win[0], fs)

    # 2D scalogram
    plot_frequency_scalogram(v_win[0], fs, freqs,
                             os.path.join(RESULTS_DIR, "scalogram_2d.png"))

    # 3D scalogram
    plot_3d_scalogram(v_win[0], fs, freqs,
                      os.path.join(RESULTS_DIR, "scalogram_3d.png"))

    # ── 6. Generate CWT Images for All Windows ───────────────────────────
    print(f"\n[6/7] Generating CWT scalogram images for {len(v_win)} windows...")
    X_images = []

    for idx in range(len(v_win)):
        img_v, _ = cwt_to_image(v_win[idx], fs)
        img_i, _ = cwt_to_image(i_win[idx], fs)

        # Stack voltage (Ch1) + current (Ch2) into 2-channel image
        cwt_combined = stack_channels(img_v, img_i)
        X_images.append(cwt_combined)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(v_win):
            print(f"  Processed {idx + 1}/{len(v_win)} windows")

    X = np.array(X_images, dtype=np.float32)
    Y = np.array(y_soc, dtype=np.float32)

    print(f"  Dataset shapes — X: {X.shape}, Y: {Y.shape}")

    # ── 7. Train/Test Split ──────────────────────────────────────────────
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SPLIT, random_state=RANDOM_SEED
    )
    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # ── 8. Build and Train CNN Model ─────────────────────────────────────
    print(f"\n[7/7] Building and training CNN model...")
    print(f"  Architecture: 2-channel CWT image → CNN → SOC")
    print(f"  Temperature branch: DISABLED (fixed 25°C)")
    print(f"  Epochs: {EPOCHS} | Batch size: {BATCH_SIZE}")

    model = build_cnn_model(image_shape=X[0].shape, use_temperature_scalar=False)
    model.summary()

    history = model.fit(
        X_train, Y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # ── 9. Evaluate Model ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  TEST SET EVALUATION")
    print("=" * 50)
    eval_results = model.evaluate(X_test, Y_test, verbose=0)

    for name, val in zip(model.metrics_names, eval_results):
        print(f"  {name.upper():>15s}: {val:.6f}")

    # ── 10. Generate Result Plots ────────────────────────────────────────
    print(f"\nGenerating result plots...")
    plot_training_history(history, os.path.join(RESULTS_DIR, "training_history.png"))

    Y_pred = model.predict(X_test, verbose=0)
    plot_predictions(Y_test, Y_pred, os.path.join(RESULTS_DIR, "predictions.png"))

    print("\n" + "=" * 70)
    print("  Pipeline completed. All results saved to:", RESULTS_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
