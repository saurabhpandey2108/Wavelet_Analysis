import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_preprocessing.preprocess import load_signals, estimate_fs, compute_soc, create_windows, temperature_scalar
from cwt_image.image_utils import cwt_to_image, stack_channels
from cnn_model.model import build_cnn_model

def plot_frequency_scalogram(v_win, fs, freqs, title="Scalogram Frequency Plot"):
    # Re-calculate CWT matrix native spacing for plotting
    from cwt_calc.cwt_utils import compute_cwt
    scales = np.arange(1, 128)
    coeffs, _ = compute_cwt(v_win, fs, scales)
    
    scalogram = np.abs(coeffs)
    scalogram = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min())

    plt.figure(figsize=(10, 6))
    plt.imshow(scalogram, aspect="auto", cmap="jet", origin="lower", 
               extent=[0, len(v_win) / fs, freqs.min(), freqs.max()])
    plt.colorbar(label="Normalized Magnitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Pseudo-frequency (Hz)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("frequency_plot.png")
    print("Saved frequency plot to frequency_plot.png")

def main():
    print("Starting Wavelet Analysis Pipeline...")
    np.random.seed(42)
    
    # 1. Load Data
    csv_path = os.path.join("dataset", "00005.csv")
    print(f"Loading {csv_path}...")
    df, voltage, current, temperature, time = load_signals(csv_path)
    
    fs = estimate_fs(time)
    print(f"Estimated Sampling Frequency: {fs:.4f} Hz")
    
    # 2. Compute SOC using Coulomb Counting
    print("Computing State of Charge (SOC)...")
    soc = compute_soc(current, time)
    
    # 3. Create Windows
    window_size = 256
    stride = 128
    
    print(f"Creating sliding windows (size={window_size}, stride={stride})...")
    v_win, y_soc = create_windows(voltage, soc, window_size, stride)
    i_win, _ = create_windows(current, soc, window_size, stride)
    t_win, _ = create_windows(temperature, soc, window_size, stride)
    
    # Generate a Frequency Plot for the first window as visualization
    print("Generating representative Frequency Plot (Scalogram)...")
    _, freqs = cwt_to_image(v_win[0], fs)
    plot_frequency_scalogram(v_win[0], fs, freqs)
    
    # 4. Generate CWT Images
    X_images = []
    T_scalars = []
    
    print(f"Generating CWT Scalograms for {len(v_win)} windows. This might take a moment...")
    for idx in range(len(v_win)):
        img_v, _ = cwt_to_image(v_win[idx], fs)
        img_i, _ = cwt_to_image(i_win[idx], fs)
        
        # Combine Voltage and Current into a 2-channel image
        cwt_combined = stack_channels(img_v, img_i)
        X_images.append(cwt_combined)
        
        # Calculate temperature scalar
        T_scalars.append(temperature_scalar(t_win[idx]))
        
        if (idx+1) % 5 == 0:
            print(f"  Processed {idx+1}/{len(v_win)} windows")

    X = np.array(X_images)
    T = np.array(T_scalars)
    Y = np.array(y_soc)
    
    # Normalize temperature
    if T.std() != 0:
        T = (T - T.mean()) / T.std()
        
    print(f"Dataset assembled. X shape: {X.shape}, T shape: {T.shape}, Y shape: {Y.shape}")
    
    # 5. Train/Test Split
    X_img_train, X_img_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=0.2, random_state=42
    )
    
    # 6. Build and Train Model
    print("Building CNN Model with Image + Temperature Scalar inputs...")
    model = build_cnn_model(image_shape=X[0].shape, use_temperature_scalar=True)
    
    print("Training Model...")
    val_split = 0.2 if len(X_img_train) >= 5 else 0.0
    history = model.fit(
        [X_img_train, T_train], Y_train,
        validation_split=val_split,
        epochs=10,
        batch_size=8,
        verbose=1
    )
    
    # 7. Evaluate Model
    print("Evaluating Model on Test Set...")
    eval_results = model.evaluate([X_img_test, T_test], Y_test, verbose=0)
    
    metric_names = model.metrics_names
    print("\n--- Test Set Evaluation ---")
    for name, val in zip(metric_names, eval_results):
        print(f"{name.upper()}: {val:.4f}")
        
    print("Pipeline Execution Completed.")

if __name__ == "__main__":
    main()
