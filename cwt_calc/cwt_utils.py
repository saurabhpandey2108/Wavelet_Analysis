"""
Continuous Wavelet Transform (CWT) — Morlet Wavelet, Pure NumPy.

No external wavelet libraries. FFT-domain linear convolution on a
reflection-padded signal, with per-scale adaptive kernel length so the
Morlet envelope decays fully at kernel boundaries.

CWT formula:
    CWT(a, b) = (1/√a) ∫ x(t) · ψ*((t - b) / a) dt
    f_pseudo = ω₀ / (2π · a · dt)

Morlet mother wavelet (ω₀ = 6):
    ψ(t) = π^(-1/4) · e^(jω₀t) · e^(-t²/2)

Boundary handling
─────────────────
Every scale's kernel extends ±4a samples around its centre (envelope
≤ 3·10⁻⁴ beyond that). At large scales the kernel is much longer than the
256-sample window, and linear FFT convolution naturally assumes zeros
outside — which produces the classic COI "bowtie" edge artefact on the
scalogram.

This implementation symmetrically reflects the signal on both sides by
`half` samples (np.pad mode='reflect') before convolving. The wavelet then
sees a continuous-looking extension at both boundaries instead of a cliff,
and the resulting CWT has far less edge contamination inside the original
window.
"""

import numpy as np


def coi_mask(n, scales):
    """Cone-of-Influence mask (True = inside COI = unreliable coefficient).

    For a Morlet wavelet at scale a, the e-folding time is τ = a·√2 samples.
    Coefficients within τ of either edge are within the COI and are edge-
    contaminated even with reflection padding (the wavelet there depends on
    the mirrored signal, not independent observations).

    Parameters
    ----------
    n : int — signal length
    scales : array_like — wavelet scales

    Returns
    -------
    mask : (len(scales), n) bool ndarray — True means coefficient inside COI.
    """
    t = np.arange(n)
    sqrt2 = np.sqrt(2.0)
    mask = np.zeros((len(scales), n), dtype=bool)
    for i, a in enumerate(scales):
        tau = a * sqrt2
        mask[i] = (t < tau) | (t > (n - 1 - tau))
    return mask


def morlet_wavelet(t, w0=6):
    """Morlet mother wavelet sampled at time points `t` (already divided by a)."""
    return (np.pi ** -0.25) * np.exp(1j * w0 * t) * np.exp(-t ** 2 / 2)


def _wavelet_kernel(scale, dt, w0=6):
    """Build a Morlet kernel long enough that the envelope has fully decayed.
    `half` is the half-width in samples; kernel length is 2·half + 1 (odd)."""
    half = int(np.ceil(4 * scale))
    kernel_len = 2 * half + 1
    tk = (np.arange(kernel_len) - half) * dt
    wavelet = morlet_wavelet(tk / scale, w0) / np.sqrt(scale)
    return wavelet, half


def compute_cwt(signal, fs, scales, w0=6):
    """CWT of a 1-D signal against `scales`, via FFT-domain convolution on
    a reflection-padded signal.

    Returns
    -------
    cwt_matrix : (len(scales), len(signal)) complex ndarray
    freqs      : (len(scales),) ndarray of pseudo-frequencies in Hz
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    dt = 1.0 / fs

    cwt_matrix = np.zeros((len(scales), n), dtype=np.complex128)
    freqs = np.zeros(len(scales))

    for i, scale in enumerate(scales):
        freqs[i] = w0 / (2 * np.pi * scale * dt)

        wavelet, half = _wavelet_kernel(scale, dt, w0=w0)

        # Reflection-pad the signal by `half` samples on each side so the
        # wavelet never sees a cliff at the original signal's boundaries.
        if half > 0:
            signal_padded = np.pad(signal, half, mode='reflect')
        else:
            signal_padded = signal

        # FFT-domain linear convolution with conj(wavelet[::-1]) = correlation
        kernel = np.conj(wavelet[::-1])
        N_full = len(signal_padded) + len(kernel) - 1
        conv = np.fft.ifft(np.fft.fft(signal_padded, N_full) *
                           np.fft.fft(kernel, N_full))

        # Extract the n samples aligned with the original signal. The full
        # linear-conv output index for "kernel centre on original position j"
        # is j + 2·half (half for kernel centering, half for the left pad).
        start = 2 * half
        cwt_matrix[i, :] = conv[start:start + n] * dt

    return cwt_matrix, freqs
