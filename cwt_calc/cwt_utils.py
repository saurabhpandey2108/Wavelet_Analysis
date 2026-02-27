import numpy as np

def morlet_wavelet(t, w0=6):
    normalization = np.pi ** (-0.25)
    return normalization * np.exp(1j * w0 * t) * np.exp(-t**2 / 2)

def compute_cwt(signal, fs, scales, w0=6):
    """
    Computes the Continuous Wavelet Transform using a Morlet Wavelet.
    signal: 1D numpy array
    fs: sampling frequency (Hz)
    scales: array of scales
    w0: central frequency of Morlet wavelet
    """
    n = len(signal)
    dt = 1.0 / fs
    # symmetric time vector centered at zero
    t = np.arange(-n//2, n//2) * dt

    cwt_matrix = np.zeros((len(scales), n), dtype=complex)
    freqs = np.zeros(len(scales))

    for i, scale in enumerate(scales):
        # Pseudo-frequency for Morlet
        freqs[i] = w0 / (2 * np.pi * scale)
        
        scaled_t = t / scale
        wavelet = morlet_wavelet(scaled_t, w0)

        # Scale normalization
        wavelet = wavelet / np.sqrt(scale)

        # Convolution
        conv = np.convolve(signal, np.conj(wavelet), mode='same')

        cwt_matrix[i, :] = conv

    return cwt_matrix, freqs

