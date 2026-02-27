# CWT-CNN Based State of Charge Estimation for EV Batteries

A research project that estimates the **State of Charge (SOC)** of Lithium-ion batteries by converting raw electrical signals into **2D Scalogram images** using the **Continuous Morlet Wavelet Transform (CWT)** and classifying them with a **Convolutional Neural Network (CNN)**.

> **Reference Paper:** *Continuous Wavelet Transform based CNN Model for EV Battery State of Charge Estimation* (see `Paper/` directory)

---

## Motivation

Accurate SOC estimation is critical for Electric Vehicle (EV) Battery Management Systems. Traditional methods like Extended Kalman Filters struggle with the non-linear electrochemical behavior of Li-ion cells. This project explores a **data-driven approach** where:

1. Raw time-domain signals (Voltage, Current) are transformed into **time-frequency scalograms** using CWT.
2. The scalograms reveal hidden frequency-domain patterns (transient responses, impedance characteristics) invisible in raw data.
3. A CNN learns these visual patterns to predict SOC — functioning as a non-invasive, real-time alternative to lab-based Electrochemical Impedance Spectroscopy (EIS).

---

## Dataset

- **Source:** NASA Battery Dataset — Cell B0005 (constant 1A discharge cycle)
- **Location:** `dataset/00005.csv`
- **Signals:** Voltage (V), Current (A), Temperature (°C), Time (s)
- **Total Samples:** 430 time steps (~93 minutes of discharge)
- **Estimated Sampling Frequency:** 0.0758 Hz (~1 reading every 13.2 seconds)

---

## Pipeline Architecture

```
dataset/00005.csv
        │
        ▼
┌─────────────────────────────┐
│   data_preprocessing/       │  Load CSV → Coulomb Counting (SOC Labels)
│   preprocess.py             │  → Sliding Windows (size=256, stride=128)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   cwt_calc/                 │  Morlet Wavelet Transform (Pure NumPy)
│   cwt_utils.py              │  → CWT Coefficient Matrix + Pseudo-frequencies
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   cwt_image/                │  Normalize → Resize to 224×224
│   image_utils.py            │  → Stack Voltage (Ch1) + Current (Ch2)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   cnn_model/                │  3-Layer CNN (Conv2D → MaxPool → GAP)
│   model.py                  │  + Temperature Scalar Input
│                             │  → Predict SOC (0.0 to 1.0)
└─────────────────────────────┘
        │
        ▼
  Evaluation Metrics: MSE, MAE, RMSE, R²
  Output: frequency_plot.png (Scalogram Visualization)
```

---

## Project Structure

```
Wavelet_Analysis/
├── dataset/
│   └── 00005.csv                  # NASA B0005 battery discharge data
├── data_preprocessing/
│   └── preprocess.py              # Signal loading, SOC calculation, windowing
├── cwt_calc/
│   └── cwt_utils.py               # Morlet wavelet & CWT (NumPy only, no pywt)
├── cwt_image/
│   └── image_utils.py             # Scalogram normalization, resizing, channel stacking
├── cnn_model/
│   └── model.py                   # CNN architecture & custom metrics (MAE, RMSE, R²)
├── Paper/
│   └── *.pdf                      # Reference research paper
├── main.py                        # End-to-end pipeline entry point
├── requirements.txt               # Python dependencies
├── frequency_plot.png             # Generated scalogram visualization
└── README.md
```

---

## Key Implementation Details

### 1. SOC Labeling — Coulomb Counting
Since SOC cannot be directly measured, it is computed by integrating current over time:

```
SOC(t) = 1.0 + (∫ I(t) dt) / Total_Capacity
```

### 2. CWT — No External Wavelet Library
The Continuous Morlet Wavelet Transform is implemented from scratch using only `numpy`. The Morlet wavelet is defined as:

```
ψ(t) = π^(-1/4) · e^(jω₀t) · e^(-t²/2)
```

where `ω₀ = 6` (central frequency). Pseudo-frequencies are calculated as `f = ω₀ / (2π · scale)`.

### 3. 2-Channel Scalogram Images
For each time window, two separate CWT scalograms are computed:
- **Channel 1:** Voltage scalogram (captures electrochemical state)
- **Channel 2:** Current scalogram (captures load/demand profile)

These are stacked into a single `224×224×2` image, allowing the CNN to learn the relationship between both signals simultaneously.

### 4. Temperature as Auxiliary Input
Mean temperature of each window is passed as a separate scalar input to the CNN (not part of the scalogram), since the same voltage/current patterns can indicate different SOC values at different temperatures.

### 5. CNN Architecture
```
Input: 224×224×2 Image + Temperature Scalar
  → Conv2D(32, 3×3) → MaxPool(2×2)
  → Conv2D(64, 3×3) → MaxPool(2×2)
  → Conv2D(128, 3×3) → GlobalAveragePooling
  → Concatenate(Temperature)
  → Dense(64, ReLU) → Dense(1, Linear)
Output: SOC prediction (0.0 to 1.0)
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error (training loss) |
| **MAE** | Mean Absolute Error — average deviation from true SOC |
| **RMSE** | Root Mean Squared Error — penalizes large prediction errors |
| **R²** | Coefficient of Determination — proportion of variance explained |

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
   - Load and preprocess the battery dataset
   - Compute SOC labels using Coulomb Counting
   - Generate CWT scalogram images for each window
   - Save a representative `frequency_plot.png`
   - Train the CNN model (80/20 train-test split)
   - Print evaluation metrics (MSE, MAE, RMSE, R²)

---

## Dependencies

- Python 3.12+
- NumPy
- Pandas
- Matplotlib
- OpenCV (`opencv-python`)
- TensorFlow / Keras
- Scikit-learn
- SciPy

---

## Future Scope

- Train on multiple discharge cycles and different battery cells for improved generalization
- Use dynamic driving profiles (UDDS, FUDS) instead of constant-current discharge
- Compare with traditional SOC estimation methods (EKF, UKF)
- Explore deeper CNN architectures (ResNet, DenseNet) for improved accuracy
