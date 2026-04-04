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
    import numpy as np

def morlet_wavelet(t, w0=6):
    return (np.pi ** -0.25) * np.exp(1j * w0 * t) * np.exp(-t**2 / 2)

def compute_cwt(signal, fs, scales, w0=6):
    signal = np.asarray(signal)
    n = len(signal)

    dt = 1.0 / fs

    # symmetric time vector
    t = (np.arange(n) - n // 2) * dt

    cwt_matrix = np.zeros((len(scales), n), dtype=np.complex64)
    import numpy as np

def morlet_wavelet(t, w0=6):
    return (np.pi ** -0.25) * np.exp(1j * w0 * t) * np.exp(-t**2 / 2)

def compute_cwt(signal, fs, scales, w0=6):
    signal = np.asarray(signal)
    n = len(signal)

    dt = 1.0 / fs

    # symmetric time vector
    t = (np.arange(n) - n // 2) * dt

    cwt_matrix = np.zeros((len(scales), n), dtype=np.complex64)
    
    freqs = np.zeros(len(scales))

    for i, scale in enumerate(scales):

        # Correct pseudo-frequency
        freqs[i] = w0 / (2 * np.pi * scale * dt)

        # Scale time axis
        scaled_t = t / scale

        # Morlet wavelet with scale normalization
        wavelet = morlet_wavelet(scaled_t, w0) / np.sqrt(scale)

        # Convolution with dt factor
        cwt_matrix[i, :] = np.convolve(
            signal,
            np.conj(wavelet[::-1]),
            mode='same'
        ) * dt

    return cwt_matrix, freqs

