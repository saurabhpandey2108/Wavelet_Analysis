"""
Data Preprocessing for Battery SOC Estimation Pipeline.

Handles loading data from different battery cycler formats (CSV and Arbin XLS),
computing SOC via Coulomb Counting, and creating sliding windows for the CWT-CNN
pipeline.
"""

import os
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
#  Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_signals_csv(csv_path, cols=None):
    """Load signals from a CSV file (e.g. NASA Battery Dataset format).

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    cols : dict, optional
        Mapping of signal names to column names. Defaults to NASA format.

    Returns
    -------
    df : DataFrame
    voltage, current, temperature, time : ndarray
    """
    df = pd.read_csv(csv_path)
    if cols is None:
        cols = {
            'voltage': 'Voltage_measured',
            'current': 'Current_measured',
            'temperature': 'Temperature_measured',
            'time': 'Time'
        }

    voltage = df[cols['voltage']].values.astype(np.float64)
    current = df[cols['current']].values.astype(np.float64)
    temperature = df[cols['temperature']].values.astype(np.float64)
    time = df[cols['time']].values.astype(np.float64)

    return df, voltage, current, temperature, time


def load_signals_xls(xls_path, sheet_name='Channel_1-008', ambient_temp=25.0):
    """Load signals from an Arbin battery cycler XLS file.

    The Arbin format stores data in a sheet named 'Channel_1-XXX' with columns:
    - Test_Time(s), Current(A), Voltage(V), Charge_Capacity(Ah),
      Discharge_Capacity(Ah), etc.

    Parameters
    ----------
    xls_path : str
        Path to the .xls file.
    sheet_name : str
        Name of the data sheet. Default 'Channel_1-008'.
    ambient_temp : float
        Ambient temperature in °C. Default 25.0.

    Returns
    -------
    df : DataFrame
    voltage, current, temperature, time : ndarray
    """
    df = pd.read_excel(xls_path, sheet_name=sheet_name, header=0,
                       engine='openpyxl')

    voltage = df['Voltage(V)'].values.astype(np.float64)
    current = df['Current(A)'].values.astype(np.float64)
    time = df['Test_Time(s)'].values.astype(np.float64)


    # Fixed ambient temperature — no temperature sensor column in Arbin data
    temperature = np.full(len(voltage), ambient_temp, dtype=np.float64)

    return df, voltage, current, temperature, time


def load_signals(path, cols=None, sheet_name='Channel_1-008', ambient_temp=25.0):
    """Auto-detect file format and load signals.

    Supports:
      - .csv  → NASA Battery Dataset format
      - .xls  → Arbin battery cycler format

    Parameters
    ----------
    path : str
        Path to the dataset file.
    cols : dict, optional
        Column mapping for CSV files.
    sheet_name : str
        Sheet name for XLS files.
    ambient_temp : float
        Ambient temperature in °C for XLS files. Default 25.0.

    Returns
    -------
    df, voltage, current, temperature, time
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xls', '.xlsx'):
        print(f"  Detected Arbin XLS format: {os.path.basename(path)}")
        return load_signals_xls(path, sheet_name=sheet_name, ambient_temp=ambient_temp)
    else:
        print(f"  Detected CSV format: {os.path.basename(path)}")
        return load_signals_csv(path, cols=cols)


# ──────────────────────────────────────────────────────────────────────────────
#  Sampling Frequency
# ──────────────────────────────────────────────────────────────────────────────

def estimate_fs(time_array):
    """Estimate sampling frequency from time stamps (Hz)."""
    return 1.0 / np.median(np.diff(time_array))


# ──────────────────────────────────────────────────────────────────────────────
#  SOC Computation
# ──────────────────────────────────────────────────────────────────────────────

# def compute_soc(current, time, initial_soc=1.0, capacity_ah=None):
#     """Calculate State of Charge (SOC) using Coulomb Counting.

#     SOC(t) = SOC_0 + ∫ I(t) dt / Q_rated

#     Sign convention (Arbin standard):
#       - Positive current → charging   (SOC increases)
#       - Negative current → discharging (SOC decreases)

#     Parameters
#     ----------
#     current : ndarray
#         Current measurements in Amperes.
#     time : ndarray
#         Time stamps in seconds.
#     initial_soc : float
#         Starting SOC (0.0 to 1.0). Use 0.5 for 50% SOC datasets.
#     capacity_ah : float or None
#         Battery rated capacity in Ah. If None, estimated from data.

#     Returns
#     -------
#     soc : ndarray
#         SOC values clipped to [0, 1].
#     """
#     dt = np.diff(time, prepend=time[0])
#     dt[0] = 0.0  # First sample has no preceding interval
#     dt_hours = dt / 3600.0

#     # Cumulative charge transferred (Ah)
#     # Positive current = charging = SOC increases
#     # Negative current = discharging = SOC decreases
#     cumulative_ah = np.cumsum(current * dt_hours)

#     if capacity_ah is None:
#         capacity_ah = abs(cumulative_ah[-1])
#         if capacity_ah == 0:
#             capacity_ah = 1.0  # Fallback to prevent division by zero

#     soc = initial_soc + (cumulative_ah / capacity_ah)
#     return np.clip(soc, 0.0, 1.0)
import numpy as np

def compute_soc(df, initial_soc=0.8):
    """
    Compute SOC using charge and discharge capacity columns.

    Parameters:
    ----------
    df : pandas DataFrame
        Must contain:
        - 'Charge_Capacity(Ah)'
        - 'Discharge_Capacity(Ah)'
    initial_soc : float
        Initial SOC (e.g., 0.8 for 80%)

    Returns:
    -------
    soc : ndarray
    """

    charge = df["Charge_Capacity(Ah)"].values.astype(float)
    discharge = df["Discharge_Capacity(Ah)"].values.astype(float)

    # Rated capacity (maximum observed discharge)
    capacity_ah = np.max(discharge)

    # Net charge (handles positive + negative current automatically)
    net_ah = charge - discharge

    # SOC calculation
    soc = initial_soc + (net_ah / capacity_ah)

    return np.clip(soc, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
#  Windowing
# ──────────────────────────────────────────────────────────────────────────────

def create_windows(signal, soc_array, window_size, stride):
    """Create sliding windows over the signal and target SOC.

    Parameters
    ----------
    signal : ndarray
        1-D input signal.
    soc_array : ndarray
        SOC values aligned with the signal.
    window_size : int
        Number of samples per window.
    stride : int
        Step size between consecutive windows.

    Returns
    -------
    windows : ndarray, shape (num_windows, window_size)
    labels : ndarray, shape (num_windows,)
        SOC label for each window (value at the end of the window).
    """
    windows = []
    labels = []
    for i in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[i:i + window_size])
        labels.append(soc_array[i + window_size - 1])
    return np.array(windows), np.array(labels)


# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────

def temperature_scalar(temp_window):
    """Return the mean temperature of a window as a scalar feature."""
    return np.mean(temp_window)


def normalize_array(x):
    """Min-max normalize an array to [0, 1]."""
    x = np.asarray(x, dtype=float)
    if np.ptp(x) == 0:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())
