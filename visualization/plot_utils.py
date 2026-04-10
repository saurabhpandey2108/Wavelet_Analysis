import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from cwt_calc.cwt_utils import compute_cwt
from config.settings import SCALES

def plot_frequency_scalogram(v_win, fs, freqs, save_path, title_suffix=""):
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
    ax.set_title(f"CWT Scalogram — Voltage Signal {title_suffix}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved 2D scalogram → {save_path}")

def plot_3d_scalogram(v_win, fs, freqs, save_path, title_suffix=""):
    """3D surface plot of the CWT scalogram."""
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

def plot_fig3_signals(us06_time, us06_i, us06_v, us06_soc, fuds_time, fuds_i, fuds_v, fuds_soc, save_path):
    """Recreate Paper Fig 3: 2x3 grid of US06 and FUDS signals."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    blue_line = '#3182bd'  # MATLAB-like blue
    
    # Top Row: US06
    axes[0, 0].plot(us06_time, us06_i, color=blue_line, linewidth=0.5)
    axes[0, 0].set_title("(a)")
    axes[0, 0].set_ylabel("Current(A)")
    
    axes[0, 1].plot(us06_time, us06_v, color=blue_line, linewidth=0.5)
    axes[0, 1].set_title("(b)")
    axes[0, 1].set_ylabel("Voltage(V)")
    
    axes[0, 2].plot(us06_time, us06_soc * 100, color=blue_line, linewidth=1.5)
    axes[0, 2].set_title("(c)")
    axes[0, 2].set_ylabel("SOC(%)")
    
    # Bottom Row: FUDS
    axes[1, 0].plot(fuds_time, fuds_i, color=blue_line, linewidth=0.5)
    axes[1, 0].set_title("(d)")
    axes[1, 0].set_ylabel("Current(A)")
    
    axes[1, 1].plot(fuds_time, fuds_v, color=blue_line, linewidth=0.5)
    axes[1, 1].set_title("(e)")
    axes[1, 1].set_ylabel("Voltage(V)")
    
    axes[1, 2].plot(fuds_time, fuds_soc * 100, color=blue_line, linewidth=1.5)
    axes[1, 2].set_title("(f)")
    axes[1, 2].set_ylabel("SOC(%)")
    
    for ax in axes.flat:
        ax.set_xlabel("Time(s)")
        ax.grid(True, alpha=0.3)
        
    # Align limits visually to the paper
    axes[0, 2].set_ylim(0, 80)
    axes[1, 2].set_ylim(0, 80)

    fig.text(0.5, 0.01, "(a) Measured current profile for US06 profile; (b) measured voltage profile for US06 profile; (c) SOC values for US06 profile; (d) measured current profile for FUDS profile; (e) measured voltage profile for FUDS profile; (f) SOC values for FUDS profile.", ha='center', fontsize=10, wrap=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved Fig 3 Signals → {save_path}")

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
    """Recreate Paper Fig 5: 3x1 grid of Predicted vs Actual SOC over time."""
    fig, axes = plt.subplots(3, 1, figsize=(6, 12))
    
    # In paper: 
    # DST: Pred(Light Blue), Act(Orange) -> Wait, paper has Pred(Light Blue), Actual(Brown-ish red). Let's use similar visual colors.
    colors = [
        {'pred': '#6baed6', 'act': '#d95f02'}, # DST
        {'pred': '#74c476', 'act': '#08519c'}, # US06
        {'pred': '#74c476', 'act': '#08519c'}, # FUDS
    ]
    
    for ax, result, col in zip(axes, test_results, colors):
        # We need time(s) for the x-axis. We calculate it.
        # test_results will now need to pass the fs, stride, window_size to reconstruct approximate time!
        time_s = result["time_axis"]
        
        ax.plot(time_s, result['y_true'] * 100, color=col['act'], linewidth=2.5, label='Actual')
        ax.plot(time_s, result['y_pred'].flatten() * 100, color=col['pred'], linewidth=1.5, label='Predicted')
        
        ax.set_ylabel("SOC(%)")
        ax.set_xlabel("Time(s)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    fig.text(0.5, 0.01, "Fig. 5. Percentage values of the battery SOC comparisons for the DST, US06, and FUDS datasets at 25 degrees Celsius.", ha='center', fontsize=10, wrap=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved Fig 5 SOC Comparison → {save_path}")
