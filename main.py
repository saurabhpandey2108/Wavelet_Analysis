"""
CWT-CNN Battery SOC Estimation Pipeline
========================================

Training:  DST @ 0°C  +  DST @ 25°C  (both dynamic profiles, Step 7 only)
Testing:   US06 @ 25°C  and  FUDS @ 25°C  (held out, different driving cycles)

Regularization:
  - Shuffled train/val split (sklearn train_test_split)
  - GaussianNoise on scalogram inputs
  - EarlyStopping on val_loss (restore best weights)
  - Temperature scalar input (meaningful now that training spans 0°C and 25°C)
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from config.settings import (
    RESULTS_DIR, EPOCHS, BATCH_SIZE, RANDOM_SEED, STRIDE,
    TRAIN_DATASETS, TEST_DATASETS,
    USE_TEMPERATURE, NOISE_STD, EARLY_STOP_PATIENCE, TEMP_NORM_DENOM,
)
from pipeline.data_pipeline import process_dataset_raw, compute_norm_stats, build_images
from cwt_image.image_utils import cwt_to_image
from cnn_model.model import build_cnn_model

from visualization.plot_utils import (
    plot_frequency_scalogram,
    plot_3d_scalogram,
    plot_fig4_training_history,
    plot_fig5_soc_comparison,
)


def main():
    print("=" * 70)
    print("  CWT-CNN Battery SOC Estimation Pipeline")
    print("  Train: DST @ 0°C + DST @ 25°C   Test: US06 & FUDS @ 25°C")
    print("=" * 70)
    np.random.seed(RANDOM_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    #  PHASE 1: RAW SCALOGRAMS (no normalization yet)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PHASE 1: RAW SCALOGRAMS — TRAIN + TEST")
    print(f"{'='*70}")

    train_raws = [process_dataset_raw(cfg, results_dir=RESULTS_DIR)
                  for cfg in TRAIN_DATASETS]
    test_raws = [process_dataset_raw(cfg, results_dir=RESULTS_DIR)
                 for cfg in TEST_DATASETS]

    # ──────────────────────────────────────────────────────────────────────
    #  PHASE 2: GLOBAL NORMALIZATION FROM TRAINING, APPLY TO ALL
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PHASE 2: GLOBAL NORMALIZATION")
    print(f"{'='*70}")

    norm_stats = compute_norm_stats(
        [r["raw_v"] for r in train_raws],
        [r["raw_i"] for r in train_raws],
    )
    print(f"  Norm stats (from training union) — "
          f"V: [{norm_stats['v_min']:.4f}, {norm_stats['v_max']:.4f}]  "
          f"I: [{norm_stats['i_min']:.4f}, {norm_stats['i_max']:.4f}]")

    X_train = np.concatenate(
        [build_images(r["raw_v"], r["raw_i"], norm_stats) for r in train_raws],
        axis=0,
    )
    Y_train = np.concatenate([r["y"] for r in train_raws], axis=0)
    T_train = np.concatenate([r["temperature"] for r in train_raws], axis=0) / TEMP_NORM_DENOM

    print(f"  Training tensors: X={X_train.shape}, Y={Y_train.shape}, T={T_train.shape}")
    for r in train_raws:
        print(f"    {r['label']}: {len(r['y'])} windows")

    test_datasets = []
    for r in test_raws:
        X = build_images(r["raw_v"], r["raw_i"], norm_stats)
        T = r["temperature"] / TEMP_NORM_DENOM
        test_datasets.append({
            "label": r["label"],
            "X": X, "Y": r["y"], "T": T,
            "fs": r["fs"],
        })
    for d in test_datasets:
        print(f"  Test {d['label']}: X={d['X'].shape}, Y={d['Y'].shape}")

    # Combined test for a single overall metric
    X_test_all = np.concatenate([d["X"] for d in test_datasets], axis=0)
    Y_test_all = np.concatenate([d["Y"] for d in test_datasets], axis=0)
    T_test_all = np.concatenate([d["T"] for d in test_datasets], axis=0)

    # Visualization off a training window (DST @ 0°C, first window)
    v_win_sample = train_raws[0]["v_win"][0]
    fs_sample = train_raws[0]["fs"]
    _, freqs = cwt_to_image(v_win_sample, fs_sample)
    plot_frequency_scalogram(v_win_sample, fs_sample, freqs,
                             os.path.join(RESULTS_DIR, "scalogram_2d_train.png"),
                             title_suffix="(DST @ 0°C, training)")
    plot_3d_scalogram(v_win_sample, fs_sample, freqs,
                      os.path.join(RESULTS_DIR, "scalogram_3d_train.png"),
                      title_suffix="(DST @ 0°C, training)")

    # ──────────────────────────────────────────────────────────────────────
    #  PHASE 3: SHUFFLED SPLIT + CNN TRAINING
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PHASE 3: CNN MODEL TRAINING")
    print(f"{'='*70}")
    print(f"  Temperature input: {USE_TEMPERATURE} | GaussianNoise std: {NOISE_STD}")
    print(f"  EarlyStopping patience: {EARLY_STOP_PATIENCE}")
    print(f"  Epochs (max): {EPOCHS} | Batch: {BATCH_SIZE}")

    X_tr, X_val, Y_tr, Y_val, T_tr, T_val = train_test_split(
        X_train, Y_train, T_train,
        test_size=0.2, shuffle=True, random_state=RANDOM_SEED,
    )
    print(f"  Train: {len(X_tr)} | Val (shuffled): {len(X_val)}")

    model = build_cnn_model(image_shape=X_train[0].shape,
                            use_temperature_scalar=USE_TEMPERATURE,
                            noise_std=NOISE_STD)
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE,
                               restore_best_weights=True, verbose=1)

    if USE_TEMPERATURE:
        history = model.fit(
            [X_tr, T_tr], Y_tr,
            validation_data=([X_val, T_val], Y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[early_stop], verbose=1,
        )
    else:
        history = model.fit(
            X_tr, Y_tr,
            validation_data=(X_val, Y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[early_stop], verbose=1,
        )

    plot_fig4_training_history(history, os.path.join(RESULTS_DIR, "Fig4_TrainingHistory.png"))

    # ──────────────────────────────────────────────────────────────────────
    #  PHASE 4: EVALUATION
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PHASE 4: MODEL EVALUATION")
    print(f"{'='*70}")

    def predict(X, T):
        return model.predict([X, T], verbose=0).flatten() if USE_TEMPERATURE \
            else model.predict(X, verbose=0).flatten()

    # Combined
    combined_pred = predict(X_test_all, T_test_all)
    combined_err = combined_pred - Y_test_all
    combined_mae = float(np.mean(np.abs(combined_err)))
    combined_rmse = float(np.sqrt(np.mean(combined_err ** 2)))
    print(f"\n  ── Combined Test (US06 + FUDS @ 25°C) ──")
    print(f"    MAE : {combined_mae:.6f}")
    print(f"    RMSE: {combined_rmse:.6f}")

    # Per-dataset
    test_results = []
    for d in test_datasets:
        label = d["label"]
        y_pred = predict(d["X"], d["T"])
        err = y_pred - d["Y"]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        print(f"\n  ── {label} ──")
        print(f"    MAE : {mae:.6f}")
        print(f"    RMSE: {rmse:.6f}")

        time_s = np.arange(len(y_pred)) * STRIDE / d["fs"]
        test_results.append({
            "label": label,
            "y_true": d["Y"],
            "y_pred": y_pred,
            "rmse": rmse,
            "mae": mae,
            "time_axis": time_s,
        })

    plot_fig5_soc_comparison(
        test_results,
        os.path.join(RESULTS_DIR, "Fig5_SOC_Comparisons.png"),
    )

    # Summary
    print(f"\n{'='*70}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Dataset':<20s} {'RMSE':>10s} {'MAE':>10s}")
    print(f"  {'─' * 40}")
    for r in test_results:
        print(f"  {r['label']:<20s} {r['rmse']:>10.6f} {r['mae']:>10.6f}")
    print(f"  {'─' * 40}")
    print(f"  {'Combined':<20s} {combined_rmse:>10.6f} {combined_mae:>10.6f}")

    print(f"\n  Results saved to: {os.path.abspath(RESULTS_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
