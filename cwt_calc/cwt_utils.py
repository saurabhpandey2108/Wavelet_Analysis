"""
Continuous Wavelet Transform (CWT) using Morlet Wavelet — Pure NumPy.

Implements the CWT from scratch without external wavelet libraries (no pywt).

Key Concepts — Shifting and Scaling in CWT:
─────────────────────────────────────────────
The CWT decomposes a signal x(t) into time-frequency space by convolving it
with shifted and scaled versions of a mother wavelet ψ(t):

    CWT(a, b) = (1/√a) ∫ x(t) · ψ*((t - b) / a) dt

where:
    a = scale parameter (controls frequency resolution)
        - Large a  → wavelet is stretched  → captures LOW  frequencies
        - Small a  → wavelet is compressed → captures HIGH frequencies
        - Pseudo-frequency: f = ω₀ / (2π · a · dt)

    b = shift/translation parameter (controls time localization)
        - Slides the wavelet along the signal to detect WHEN features occur
        - Each value of b gives a coefficient at that time position

    1/√a = normalization factor
        - Ensures energy is preserved across scales
        - Without this, larger scales would have artificially higher coefficients

The Morlet wavelet is defined as:
    ψ(t) = π^(-1/4) · e^(jω₀t) · e^(-t²/2)

where ω₀ = 6 is the central angular frequency (standard choice for good
time-frequency trade-off).

The output CWT matrix has shape (num_scales, signal_length):
    - Rows → different scales (frequencies)
    - Columns → different shifts (time positions)
    - |CWT|² gives the scalogram (energy distribution in time-frequency plane)
"""

import numpy as np


def morlet_wavelet(t, w0=6):
    """Morlet mother wavelet.

    Parameters
    ----------
    t : ndarray
        Time array (already scaled by 1/a for the desired scale).
    w0 : float
        Central angular frequency. Default 6 gives ~1 Hz centre frequency
        at scale = 1, and ensures the admissibility condition is satisfied.

    Returns
    -------
    ndarray (complex)
        Wavelet values at each time point.
    """
    return (np.pi ** -0.25) * np.exp(1j * w0 * t) * np.exp(-t**2 / 2)


def compute_cwt(signal, fs, scales, w0=6):
    """Compute the Continuous Wavelet Transform of a 1-D signal.

    Uses convolution in the time domain (not FFT) for conceptual clarity.

    Parameters
    ----------
    signal : array_like
        Input 1-D signal.
    fs : float
        Sampling frequency in Hz.
    scales : array_like
        Array of wavelet scales to evaluate. Each scale `a` maps to a
        pseudo-frequency  f = w0 / (2π · a · dt).
    w0 : float
        Central angular frequency of the Morlet wavelet.

    Returns
    -------
    cwt_matrix : ndarray, shape (len(scales), len(signal))
        Complex CWT coefficient matrix.
    freqs : ndarray, shape (len(scales),)
        Pseudo-frequencies (Hz) corresponding to each scale.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    dt = 1.0 / fs

    # Symmetric time vector centred at zero for the wavelet kernel
    t = (np.arange(n) - n // 2) * dt

    cwt_matrix = np.zeros((len(scales), n), dtype=np.complex128)
    freqs = np.zeros(len(scales))

    for i, scale in enumerate(scales):
        # Pseudo-frequency for this scale
        freqs[i] = w0 / (2 * np.pi * scale * dt)

        # Scale (compress/stretch) the time axis
        scaled_t = t / scale

        # Evaluate the wavelet at the scaled time points,
        # with 1/√a energy normalization
        wavelet = morlet_wavelet(scaled_t, w0) / np.sqrt(scale)

        # Convolution implements the shift (translation) parameter:
        # for each output sample b, it computes the inner product of
        # x(t) with the wavelet centred at b.
        cwt_matrix[i, :] = np.convolve(
            signal,
            np.conj(wavelet[::-1]),
            mode='same'
        ) * dt

    return cwt_matrix, freqs
