# Wavelet Analysis for EV Battery SOC Estimation

This project implements an end-to-end pipeline to estimate the State of Charge (SOC) of Electric Vehicle (EV) batteries using a Continuous Wavelet Transform (CWT) based Convolutional Neural Network (CNN).

## Key Features & Constraints
* **No `pywavelets` or `scipy.signal` library**: The Continuous Morlet Wavelet Transform is calculated strictly using foundational mathematical equations and `numpy` arrays.
* **Coulomb Counting**: The labels for the CNN (State of Charge) are computed directly from the current over time using Coulomb counting.
* **Scalogram Image Generation**: 1D Battery voltage and current signals are mapped sequentially into 2D scalogram images representing their time-frequency combinations.
* **Multi-input CNN Model**: A custom Keras-based CNN that takes in a 2-channel Image (Voltage CWT + Current CWT) alongside a scalar tensor representing the average temperature of that window.

## Project Structure

* `dataset/`: Contains raw battery cycle CSV logs (e.g., NASA B0005 structure).
* `data_preprocessing/`: Scripts for parsing signals, calculating actual SOC from current, and generating sliding windows.
* `cwt_calc/`: The core mathematical implementation of the Morlet Wavelet and Continuous Wavelet Transform (`numpy` only).
* `cwt_image/`: Scripts to normalize and convert CWT coefficients into resized 2-channel images for the CNN.
* `cnn_model/`: Definition of the Keras CNN architecture and custom evaluation metrics (RMSE, MAE, R-squared).
* `main.py`: The entry point that ties the entire pipeline together.

## How to Run

1. Ensure dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the full pipeline:
   ```bash
   python main.py
   ```

This will automatically load the dataset, process sliding windows, generate CWT images, train the CNN model, print evaluation metrics, and save a representative `frequency_plot.png`.
