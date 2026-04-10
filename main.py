"""
CWT-CNN Battery SOC Estimation Pipeline
========================================

End-to-end pipeline for estimating the State of Charge (SOC) of Li-ion batteries
using Continuous Wavelet Transform scalograms and a Convolutional Neural Network.

Paper methodology:
  - TRAINING:  DST cycle at 0°C ambient   (02_24_2016_SP20-2_0C_DST_80SOC.xls)
  - TESTING:   DST, FUDS, US06 at 25°C    (~10,000 combined test samples)
"""

import os
import numpy as np

# Config and Utilities
from config.settings import RESULTS_DIR, EPOCHS, BATCH_SIZE, RANDOM_SEED, TRAIN_DATA, TEST_DATA
from pipeline.data_pipeline import process_dataset
from cwt_image.image_utils import cwt_to_image
from cnn_model.model import build_cnn_model

# Visualizations
from visualization.plot_utils import (
    plot_frequency_scalogram,
    plot_3d_scalogram,
    plot_fig3_signals,
    plot_fig4_training_history,
    plot_fig5_soc_comparison
)

# ──────────────────────────────────────────────────────────────────────────────
#  Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  CWT-CNN Battery SOC Estimation Pipeline")
    print("  Paper: CWT-based CNN Model for EV Battery SOC Estimation")
    print("  ─────────────────────────────────────────────────────")
    print("  Training:  DST cycle @ 0°C ambient")
    print("  Testing:   DST, US06, FUDS @ 25°C ambient")
    print("=" * 70)
    np.random.seed(RANDOM_SEED)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1: TRAINING DATA PREPARATION (DST @ 0°C)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  PHASE 1: TRAINING DATA — DST @ 0°C")
    print(f"{'='*70}")

    X_train, Y_train, fs_train, v_win_train, _, _, _, _ = \
        process_dataset(TRAIN_DATA, results_dir=RESULTS_DIR)

    print(f"\n  Training dataset shape: X={X_train.shape}, Y={Y_train.shape}")

    # Generate scalogram visualizations from training data
    print(f"\n  Generating scalogram visualizations (training data)...")
    _, freqs = cwt_to_image(v_win_train[0], fs_train)
    plot_frequency_scalogram(v_win_train[0], fs_train, freqs,
                             os.path.join(RESULTS_DIR, "scalogram_2d_train.png"),
                             title_suffix="(DST @ 0°C)")
    plot_3d_scalogram(v_win_train[0], fs_train, freqs,
                      os.path.join(RESULTS_DIR, "scalogram_3d_train.png"),
                      title_suffix="(DST @ 0°C)")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2: TEST DATA PREPARATION (DST, US06, FUDS @ 25°C)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  PHASE 2: TEST DATA — DST, US06, FUDS @ 25°C")
    print(f"{'='*70}")

    test_datasets = []
    # Collect raw data separately for Fig 3
    us06_raw, fuds_raw = None, None
    
    for test_config in TEST_DATA:
        X_test_i, Y_test_i, fs_test, _, v_raw, i_raw, t_raw, soc_raw = process_dataset(
            test_config, results_dir=RESULTS_DIR
        )
        test_datasets.append({
            "label": test_config["label"],
            "X": X_test_i,
            "Y": Y_test_i,
            "fs": fs_test
        })
        
        # Save raw data for US06 and FUDS to plot Fig 3
        if "US06" in test_config["label"]:
            us06_raw = (t_raw, i_raw, v_raw, soc_raw)
        elif "FUDS" in test_config["label"]:
            fuds_raw = (t_raw, i_raw, v_raw, soc_raw)

    # ── Recreate Fig 3 (Raw Signals US06 & FUDS) ──
    if us06_raw and fuds_raw:
        plot_fig3_signals(
            us06_raw[0], us06_raw[1], us06_raw[2], us06_raw[3],
            fuds_raw[0], fuds_raw[1], fuds_raw[2], fuds_raw[3],
            os.path.join(RESULTS_DIR, "Fig3_RawSignals.png")
        )

    # Combine all test data
    X_test_all = np.concatenate([d["X"] for d in test_datasets], axis=0)
    Y_test_all = np.concatenate([d["Y"] for d in test_datasets], axis=0)

    print(f"\n  Combined test dataset: X={X_test_all.shape}, Y={Y_test_all.shape}")
    print(f"  Per-dataset breakdown:")
    for d in test_datasets:
        print(f"    {d['label']}: {len(d['X'])} samples")
    print(f"  Total test samples: {len(X_test_all)}")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 3: BUILD AND TRAIN CNN MODEL
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  PHASE 3: CNN MODEL TRAINING")
    print(f"{'='*70}")
    print(f"  Architecture: 3×(Conv2D→BN→ReLU→MaxPool) → Dropout → FC → Regression")
    print(f"  Optimizer: Adam (lr=0.0001, clipnorm=1.0) with He Normal Initialization")
    print(f"  Epochs: {EPOCHS} | Batch size: {BATCH_SIZE}")
    print(f"  Training samples: {len(X_train)} | Validation: 20% of training")

    model = build_cnn_model(image_shape=X_train[0].shape, use_temperature_scalar=False)
    model.summary()

    history = model.fit(
        X_train, Y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Save training history plot (Fig 4)
    plot_fig4_training_history(history, os.path.join(RESULTS_DIR, "Fig4_TrainingHistory.png"))

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 4: EVALUATION — PER-DATASET AND COMBINED
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  PHASE 4: MODEL EVALUATION")
    print(f"{'='*70}")

    # ── 4a. Combined test evaluation ─────────────────────────────────────
    print("\n  ── Combined Test Set (DST + US06 + FUDS @ 25°C) ──")
    eval_combined = model.evaluate(X_test_all, Y_test_all, verbose=0)
    for name, val in zip(model.metrics_names, eval_combined):
        print(f"    {name.upper():>15s}: {val:.6f}")

    Y_pred_all = model.predict(X_test_all, verbose=0)

    # ── 4b. Per-dataset evaluation ───────────────────────────────────────
    test_results = []
    for d in test_datasets:
        label = d["label"]
        safe_label = label.replace(" ", "_").replace("°", "").replace("@", "at")
        print(f"\n  ── {label} ──")
        eval_res = model.evaluate(d["X"], d["Y"], verbose=0)
        for name, val in zip(model.metrics_names, eval_res):
            print(f"    {name.upper():>15s}: {val:.6f}")

        Y_pred_i = model.predict(d["X"], verbose=0)

        # Completely bypass Keras dictionary metric parsing bugs (which showed MAE as 0)
        # and calculate the true MAE and RMSE manually using NumPy:
        error = Y_pred_i.flatten() - d["Y"]
        mae_val = float(np.mean(np.abs(error)))
        rmse_val = float(np.sqrt(np.mean(np.square(error))))

        # Calculate time axis in seconds for Fig 5
        # Each window is offset by STRIDE samples
        from config.settings import STRIDE
        time_s = np.arange(len(Y_pred_i)) * STRIDE / d['fs']

        test_results.append({
            "label": label,
            "y_true": d["Y"],
            "y_pred": Y_pred_i,
            "rmse": rmse_val,
            "mae": mae_val,
            "time_axis": time_s
        })

    # ── 4c. Combined SOC comparison plot (Fig 5) ─────────────────────────
    plot_fig5_soc_comparison(
        test_results,
        os.path.join(RESULTS_DIR, "Fig5_SOC_Comparisons.png")
    )

    # ── 4d. Summary table ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Dataset':<20s} {'RMSE':>10s} {'MAE':>10s}")
    print(f"  {'─' * 40}")
    for r in test_results:
        print(f"  {r['label']:<20s} {r['rmse']:>10.6f} {r['mae']:>10.6f}")
    print(f"  {'─' * 40}")

    # Calculate combined metrics manually to avoid metric parsing issues
    combined_error = Y_pred_all.flatten() - Y_test_all
    combined_mae = float(np.mean(np.abs(combined_error)))
    combined_rmse = float(np.sqrt(np.mean(np.square(combined_error))))
            
    print(f"  {'Combined':<20s} {combined_rmse:>10.6f} {combined_mae:>10.6f}")

    print(f"\n  All results saved to: {os.path.abspath(RESULTS_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
