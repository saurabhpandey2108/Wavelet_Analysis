# CWT-CNN Based State of Charge Estimation for EV Batteries

A research project that estimates the **State of Charge (SOC)** of Lithium-ion batteries by converting raw electrical signals into **2D Scalogram images** using the **Continuous Morlet Wavelet Transform (CWT)** and predicting SOC with a **Convolutional Neural Network (CNN)**.

> **Reference Paper:** *Continuous Wavelet Transform based CNN Model for EV Battery State of Charge Estimation* (see `Paper/` directory)

---

## Motivation

Accurate SOC estimation is critical for Electric Vehicle (EV) Battery Management Systems. Traditional methods like Extended Kalman Filters struggle with the non-linear electrochemical behavior of Li-ion cells. This project explores a **data-driven approach** where:

1. Raw time-domain signals (Voltage, Current) are transformed into **time-frequency scalograms** using CWT.
2. The scalograms reveal hidden frequency-domain patterns (transient responses, impedance characteristics) invisible in raw data.
3. A CNN learns these visual patterns to predict SOC — functioning as a non-invasive, real-time alternative to lab-based Electrochemical Impedance Spectroscopy (EIS).

---

## Dataset

- **Source:** Arbin Battery Cycler — SP20 Cell
- **Profile:** Dynamic Stress Test (DST) at **25°C fixed ambient temperature**
- **File:** `dataset/11_05_2015_SP20-2_DST_50SOC.xls`
- **Initial SOC:** 50% (starts half-charged, charged to 100%, then DST discharge)
- **Signals:** Voltage (V), Current (A), Test Time (s)
- **Total Samples:** 9,501 data points (~9.67 hours)
- **Sampling Frequency:** ~0.985 Hz (~1 reading per second)
- **Current Range:** -4.0 A to +2.0 A (bidirectional — charge & discharge)
- **Battery Capacity:** 2.0 Ah (rated)

---

## Pipeline Architecture

```
dataset/11_05_2015_SP20-2_DST_50SOC.xls
        │
        ▼
┌─────────────────────────────────┐
│   data_preprocessing/           │  Load XLS (Arbin format)
│   preprocess.py                 │  → Coulomb Counting (SOC, initial=50%)
│                                 │  → Sliding Windows (size=256, stride=128)
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│   cwt_calc/                     │  Morlet Wavelet Transform (Pure NumPy)
│   cwt_utils.py                  │  → CWT Coefficient Matrix + Pseudo-frequencies
│                                 │  → Scaling & Shifting documented in code
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│   cwt_image/                    │  Normalize (log1p) → Resize to 224×224
│   image_utils.py                │  → Stack Voltage (Ch1) + Current (Ch2)
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│   cnn_model/                    │  3-Layer CNN (Conv2D → MaxPool → GAP)
│   model.py                      │  → Predict SOC (0.0 to 1.0)
│                                 │  (No temperature branch — fixed 25°C)
└─────────────────────────────────┘
        │
        ▼
  Evaluation Metrics: MSE, MAE, RMSE, R²
  Output: results/ (plots, training curves, predictions)
```

---

## Project Structure

```
Wavelet_Analysis/
├── dataset/
│   ├── 11_05_2015_SP20-2_DST_50SOC.xls   # Arbin DST data (SP20, 25°C, 50% SOC)
│   └── 00005.csv                           # Legacy NASA B0005 dataset
├── data_preprocessing/
│   └── preprocess.py              # Signal loading (CSV/XLS), SOC, windowing
├── cwt_calc/
│   └── cwt_utils.py               # Morlet wavelet & CWT (NumPy only, no pywt)
├── cwt_image/
│   └── image_utils.py             # Scalogram normalization, resizing, stacking
├── cnn_model/
│   └── model.py                   # CNN architecture & custom metrics
├── results/                       # Generated plots and visualizations
│   ├── raw_signals.png            # Voltage/Current time series
│   ├── soc_profile.png            # Coulomb Counting SOC curve
│   ├── scalogram_2d.png           # 2D CWT scalogram (frequency plot)
│   ├── scalogram_3d.png           # 3D CWT scalogram surface
│   ├── training_history.png       # Loss/metric training curves
│   └── predictions.png            # Predicted vs Actual SOC scatter
├── Paper/
│   └── *.pdf                      # Reference research paper
├── main.py                        # End-to-end pipeline entry point
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Key Implementation Details

### 1. SOC Labeling — Coulomb Counting
SOC is computed by integrating current over time, starting from 50% SOC:

```
SOC(t) = 0.5 + (∫ I(t) dt) / Q_rated
```

where `Q_rated = 2.0 Ah`. Positive current → charging (SOC↑), negative → discharging (SOC↓).

### 2. CWT — Scaling and Shifting Parameters

The CWT decomposes a signal using **two parameters**:

```
CWT(a, b) = (1/√a) ∫ x(t) · ψ*((t - b) / a) dt
```

| Parameter | Name | Effect |
|-----------|------|--------|
| **a** (scale) | Scaling | Stretches/compresses the wavelet. Large a → low freq, small a → high freq |
| **b** (shift) | Translation | Slides the wavelet along the signal to localize features in time |
| **1/√a** | Normalization | Preserves energy across scales so coefficients are comparable |

The Morlet wavelet `ψ(t) = π^(-1/4) · e^(jω₀t) · e^(-t²/2)` with `ω₀ = 6` provides optimal time-frequency trade-off. Pseudo-frequencies: `f = ω₀ / (2π · a · dt)`.

### 3. 2-Channel Scalogram Images
For each window:
- **Channel 1:** Voltage CWT scalogram (electrochemical state)
- **Channel 2:** Current CWT scalogram (load profile)

Stacked into `224×224×2` images for the CNN.

### 4. Temperature
The DST dataset operates at **fixed 25°C ambient**. Since temperature doesn't vary, the temperature scalar branch is **disabled** — the CNN uses only the 2-channel CWT image.

### 5. CNN Architecture
```
Input: 224×224×2 Image
  → Conv2D(32, 3×3) → MaxPool(2×2)
  → Conv2D(64, 3×3) → MaxPool(2×2)
  → Conv2D(128, 3×3) → GlobalAveragePooling
  → Dense(64, ReLU) → Dense(1, Linear)
Output: SOC prediction (0.0 to 1.0)
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error (training loss) |
| **MAE** | Mean Absolute Error — average deviation from true SOC |
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **R²** | Coefficient of Determination — variance explained |

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the full pipeline:**
   ```bash
   python main.py
   ```

   This will:
   - Load and preprocess the Arbin DST dataset
   - Compute SOC labels via Coulomb Counting (initial SOC = 50%)
   - Generate 2D and 3D CWT scalogram visualizations
   - Train the CNN model (80/20 split, 30 epochs)
   - Evaluate and save all results to `results/`

---

## Dependencies

- Python 3.12+
- NumPy
- Pandas
- Matplotlib
- OpenCV (`opencv-python`)
- TensorFlow / Keras
- Scikit-learn
- openpyxl (for Arbin XLS files)

---

## Future Scope

- Train on multiple DST cycles and temperature conditions for generalization
- Compare DST, UDDS, and FUDS dynamic profiles
- Implement deeper architectures (ResNet, DenseNet) for improved accuracy
- Compare with traditional SOC methods (EKF, UKF)
- Add early stopping and learning rate scheduling for better convergence
