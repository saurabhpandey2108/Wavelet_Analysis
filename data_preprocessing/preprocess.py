"""
Data Preprocessing for Battery SOC Estimation Pipeline.

Handles loading data from Arbin XLS battery-cycler files, isolating the
dynamic driving profile (Step_Index == 7), computing SOC from the rebased
cumulative Charge/Discharge capacity columns, and creating sliding windows
for the CWT-CNN pipeline.
"""

import os
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
#  Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_signals_csv(csv_path, cols=None):
    """Load signals from a CSV file (e.g. NASA Battery Dataset format)."""
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


def load_signals_xls(xls_path, sheet_name='Channel_1-008', ambient_temp=25.0,
                     step_filter=7):
    """Load signals from an Arbin XLS file, keeping only the dynamic driving profile.

    Arbin test schedules typically contain several setup steps (charge to 100%,
    CV hold, rest, reference discharge) before the actual DST/US06/FUDS profile
    in Step_Index == 7. Those setup steps pollute Coulomb-counting SOC because
    the Charge_Capacity/Discharge_Capacity columns accumulate through them.

    By filtering to `step_filter` (default 7) we isolate the dynamic profile
    and rebase Test_Time to start at 0. compute_soc() then rebases the capacity
    columns to zero at the first retained row, so SOC(t=0) = initial_soc.

    Parameters
    ----------
    xls_path : str
    sheet_name : str
    ambient_temp : float
    step_filter : int or None
        Step_Index to retain. If None, the whole file is returned (legacy behaviour).

    Returns
    -------
    df, voltage, current, temperature, time
    """
    df = pd.read_excel(xls_path, sheet_name=sheet_name, header=0,
                       engine='openpyxl')

    if step_filter is not None and 'Step_Index' in df.columns:
        df = df[df['Step_Index'] == step_filter].reset_index(drop=True)

    voltage = df['Voltage(V)'].values.astype(np.float64)
    current = df['Current(A)'].values.astype(np.float64)
    time = df['Test_Time(s)'].values.astype(np.float64)
    if len(time) > 0:
        time = time - time[0]  # rebase dynamic-profile time to 0

    temperature = np.full(len(voltage), ambient_temp, dtype=np.float64)

    return df, voltage, current, temperature, time


def load_signals(path, cols=None, sheet_name='Channel_1-008', ambient_temp=25.0,
                 step_filter=7):
    """Auto-detect file format and load signals. XLS files are filtered to
    `step_filter` (default Step 7 = dynamic profile). CSV files bypass filtering."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xls', '.xlsx'):
        print(f"  Detected Arbin XLS format: {os.path.basename(path)}")
        return load_signals_xls(path, sheet_name=sheet_name,
                                ambient_temp=ambient_temp, step_filter=step_filter)
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

def compute_soc(df, initial_soc=0.8, capacity_ah=2.0):
    """Compute SOC from Arbin's cumulative Charge/Discharge capacity columns.

        SOC(t) = SOC_0 + ((Charge(t) - Charge(0)) - (Discharge(t) - Discharge(0))) / Q_rated

    The capacity columns are rebased to zero at the first row of `df`. When the
    caller has already sliced to the dynamic profile (Step_Index == 7), this
    gives SOC(t=0) = initial_soc and lets the coulomb balance track only the
    Ah flowing during the dynamic profile, not during setup charging.
    """
    charge = df["Charge_Capacity(Ah)"].values.astype(float)
    discharge = df["Discharge_Capacity(Ah)"].values.astype(float)

    # Rebase to zero at first retained sample
    charge = charge - charge[0]
    discharge = discharge - discharge[0]

    net_ah = charge - discharge
    soc = initial_soc + (net_ah / capacity_ah)

    return np.clip(soc, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
#  Windowing
# ──────────────────────────────────────────────────────────────────────────────

def create_windows(signal, soc_array, window_size, stride):
    """Sliding windows with label = SOC at the end of each window."""
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
