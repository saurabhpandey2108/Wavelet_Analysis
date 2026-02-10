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


def create_windows(signal, window_size, stride):
    windows = []
    for i in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[i:i + window_size])
    return np.array(windows)


def temperature_scalar(temp_window):
    return np.mean(temp_window)


def normalize_array(x):
    x = np.asarray(x, dtype=float)
    if np.ptp(x) == 0:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())
