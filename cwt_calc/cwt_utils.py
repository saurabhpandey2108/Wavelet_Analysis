import numpy as np
import pywt


def compute_cwt(signal, fs, wavelet='morl', scales=None):
    """Compute continuous wavelet transform using PyWavelets.

    Returns coeffs (scales x time) and corresponding pseudo-frequencies.
    Default wavelet is Morlet ('morl').
    """
    if scales is None:
        scales = np.arange(1, 128)

    coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1.0 / fs)
    return coeffs, freqs
