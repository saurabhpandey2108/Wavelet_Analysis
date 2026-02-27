import numpy as np
import pandas as pd


def load_signals(csv_path, cols=None):
    """Load CSV and return signals as numpy arrays.

    cols: optional dict mapping expected names to actual column names, e.g.
        {'voltage': 'Voltage_measured', 'current': 'Current_measured',
         'temperature': 'Temperature_measured', 'time': 'Time'}
    """
    df = pd.read_csv(csv_path)
    if cols is None:
        cols = {
            'voltage': 'Voltage_measured',
            'current': 'Current_measured',
            'temperature': 'Temperature_measured',
            'time': 'Time'
        }

    voltage = df[cols['voltage']].values
    current = df[cols['current']].values
    temperature = df[cols['temperature']].values
    time = df[cols['time']].values

    return df, voltage, current, temperature, time


def estimate_fs(time_array):
    """Estimate sampling frequency from time stamps (Hz)."""
    return 1.0 / np.median(np.diff(time_array))


def compute_soc(current, time, initial_soc=1.0, capacity_ah=None):
    """
    Calculate State of Charge (SOC) using Coulomb Counting.
    current: array of current measurements (A)
    time: array of time stamps (seconds)
    initial_soc: starting SOC (0.0 to 1.0)
    capacity_ah: battery capacity in Ampere-hours. If None, it's estimated from total discharged capacity.
    """
    dt = np.diff(time, prepend=0)
    # Convert seconds to hours for Ah
    dt_hours = dt / 3600.0
    
    # Cumulative capacity (Ah)
    discharged_capacity = np.cumsum(current * dt_hours)
    
    if capacity_ah is None:
        # Assuming the entire dataset captures a full 100% to 0% discharge
        capacity_ah = abs(discharged_capacity[-1])
        if capacity_ah == 0:
            capacity_ah = 1.0 # fallback to prevent division by zero
            
    soc = initial_soc + (discharged_capacity / capacity_ah)
    return np.clip(soc, 0.0, 1.0)

def create_windows(signal, soc_array, window_size, stride):
    """
    Create sliding windows over the signal and the target SOC.
    Returns the windows and the SOC label corresponding to the *end* of each window.
    """
    windows = []
    labels = []
    for i in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[i:i + window_size])
        labels.append(soc_array[i + window_size - 1])
    return np.array(windows), np.array(labels)

def temperature_scalar(temp_window):
    return np.mean(temp_window)


def normalize_array(x):
    x = np.asarray(x, dtype=float)
    if np.ptp(x) == 0:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())
