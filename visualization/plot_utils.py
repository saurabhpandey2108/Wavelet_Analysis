import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from cwt_calc.cwt_utils import compute_cwt, coi_mask
from cwt_image.image_utils import preprocess_window
from config.settings import SCALES


def _scalogram_for_display(v_win, fs):
    """Preprocess window, compute CWT magnitude in dB, build COI-masked array.

    Returns
    -------
    scalogram : np.ma.MaskedArray of shape (len(SCALES), len(v_win)) — dB
                magnitude, with COI regions masked.
    f         : (len(SCALES),) frequencies ascending (low → high).
    coi_line  : (len(v_win),) highest reliable pseudo-freq at each time-point;
                used to draw a boundary line on the plot.
    """
    x = preprocess_window(v_win)
    coeffs, freqs = compute_cwt(x, fs, SCALES)
    n = len(v_win)

    mag = np.abs(coeffs)
    db = 20.0 * np.log10(mag + 1e-10)

    # Build COI boolean mask then flip to ascending frequency order
    mask = coi_mask(n, SCALES)

    order = np.argsort(freqs)
    f_sorted = freqs[order]
    db_sorted = db[order]
    mask_sorted = mask[order]

    scalogram = np.ma.array(db_sorted, mask=mask_sorted)

    # COI boundary: at each time t, find the lowest scale (= highest freq)
    # that is still inside the COI, and report its pseudo-frequency. The
    # region ABOVE this line on the plot is outside COI (reliable).
    # In practice we just draw the reliability envelope as two cones:
    # reliable between t = a·√2 and t = n − 1 − a·√2 at scale a.
    # We build coi_freq(t) = highest freq whose scale still has t inside COI.
    t = np.arange(n)
    # For each time t, find the largest scale a such that t < a·√2 or t > n-1-a·√2
    # i.e. a > t/√2 (left side) or a > (n-1-t)/√2 (right side).
    # The smallest such a is min(t, n-1-t)/√2; at smaller scales t is outside COI.
    dist_to_edge = np.minimum(t, n - 1 - t)
    a_coi = dist_to_edge / np.sqrt(2.0)
    # Corresponding pseudo-freq at that scale
    w0 = 6.0
    dt = 1.0 / fs
    coi_freq = w0 / (2 * np.pi * np.where(a_coi < 1e-9, 1e-9, a_coi) * dt)
    coi_freq = np.clip(coi_freq, f_sorted.min(), f_sorted.max())

    return scalogram, f_sorted, coi_freq


def plot_frequency_scalogram(v_win, fs, freqs, save_path, title_suffix=""):
    """2D scalogram — dB magnitude, log-scaled freq axis, COI shaded out."""
    scalogram, f, coi_freq = _scalogram_for_display(v_win, fs)

    # Per-plot normalize unmasked values for visual contrast
    valid = scalogram.compressed()
    lo, hi = (np.percentile(valid, 1), np.percentile(valid, 99)) if valid.size else (0, 1)
    norm = np.ma.clip((scalogram - lo) / max(hi - lo, 1e-10), 0.0, 1.0)

    time = np.arange(len(v_win)) / fs

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad(color='lightgray', alpha=0.8)
    mesh = ax.pcolormesh(time, f, norm, cmap=cmap, shading='auto')
    fig.colorbar(mesh, ax=ax, label="Normalized dB Magnitude")

    # COI boundary lines (anything BELOW these curves is inside COI)
    ax.plot(time, coi_freq, color='white', linewidth=1.2, linestyle='--', alpha=0.9)
    # Hatched shading for the COI region itself
    ax.fill_between(time, f.min(), coi_freq, color='white', alpha=0.15, hatch='///',
                    edgecolor='white', linewidth=0)

    ax.set_yscale('log')
    ax.set_ylim(f.min(), f.max())
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Pseudo-frequency (Hz)", fontsize=12)
    ax.set_title(f"CWT Scalogram — Voltage Signal {title_suffix}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved 2D scalogram → {save_path}")


def plot_3d_scalogram(v_win, fs, freqs, save_path, title_suffix=""):
    """3D surface plot of the CWT dB scalogram with COI masked out."""
    scalogram, f, _ = _scalogram_for_display(v_win, fs)

    valid = scalogram.compressed()
    lo, hi = (np.percentile(valid, 1), np.percentile(valid, 99)) if valid.size else (0, 1)
    norm = np.ma.clip((scalogram - lo) / max(hi - lo, 1e-10), 0.0, 1.0)
    # Fill masked values with NaN so they don't plot
    norm_filled = np.ma.filled(norm, np.nan)

    time_axis = np.arange(len(v_win)) / fs
    T, F = np.meshgrid(time_axis, f)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(T, F, norm_filled, cmap='jet', edgecolor='none',
                           alpha=0.9, rstride=2, cstride=2)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Normalized dB Magnitude")

    ax.set_xlabel("Time (s)", fontsize=11, labelpad=10)
    ax.set_ylabel("Pseudo-frequency (Hz)", fontsize=11, labelpad=10)
    ax.set_zlabel("Normalized Magnitude", fontsize=11, labelpad=10)
    ax.set_title(f"3D CWT Scalogram — Voltage Signal {title_suffix}", fontsize=14, pad=20)
    ax.view_init(elev=30, azim=225)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved 3D scalogram → {save_path}")

def plot_soc_profile(soc, time, save_path, title="SOC Profile"):
    """Plot the computed SOC profile over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time / 3600, soc, color='#2196F3', linewidth=1.5)
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("SOC", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved SOC profile → {save_path}")

def plot_voltage_current(voltage, current, time, save_path, title="Raw Signals"):
    """Plot raw voltage and current signals."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(time / 3600, voltage, color='#FF5722', linewidth=0.8)
    ax1.set_ylabel("Voltage (V)", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(time / 3600, current, color='#4CAF50', linewidth=0.8)
    ax2.set_xlabel("Time (hours)", fontsize=12)
    ax2.set_ylabel("Current (A)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved V/I plot → {save_path}")

def plot_fig4_training_history(history, save_path):
    """Recreate Paper Fig 4: 2x1 grid of RMSE and Loss over Epochs."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    
    blue_line = '#3182bd'
    purple_line = '#756bb1'
    
    epochs = np.arange(1, len(history.history['loss']) + 1)
    
    ax1.plot(epochs, history.history['rmse'], color=blue_line, linewidth=2)
    ax1.set_ylabel("RMSE")
    ax1.set_xlabel("Epochs") # Technically paper says Iterations, but keras yields epochs
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    ax2.plot(epochs, history.history['loss'], color=purple_line, linewidth=2)
    ax2.set_ylabel("Loss function")
    ax2.set_xlabel("Epochs")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    fig.text(0.5, 0.01, "Fig. 4. Training RMSE values and loss function variation over 30 epochs", ha='center', fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved Fig 4 Training History → {save_path}")

def plot_fig5_soc_comparison(test_results, save_path):
    """Predicted vs Actual SOC per test set. Grid size adapts to len(test_results)."""
    n = max(len(test_results), 1)
    fig, axes = plt.subplots(n, 1, figsize=(6, 4 * n), squeeze=False)
    axes = axes.flatten()

    pred_color = '#6baed6'
    act_color = '#d95f02'

    for ax, result in zip(axes, test_results):
        time_s = result["time_axis"]
        ax.plot(time_s, result['y_true'] * 100, color=act_color, linewidth=2.5, label='Actual')
        ax.plot(time_s, np.asarray(result['y_pred']).flatten() * 100,
                color=pred_color, linewidth=1.5, label='Predicted')
        ax.set_title(f"{result['label']} — RMSE {result['rmse']:.4f}, MAE {result['mae']:.4f}")
        ax.set_ylabel("SOC(%)")
        ax.set_xlabel("Time(s)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved Fig 5 SOC Comparison → {save_path}")
