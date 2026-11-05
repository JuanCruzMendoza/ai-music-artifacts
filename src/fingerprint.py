"""Huella espectral ("fakeprint"), replicando compute_fakeprints.py de Deezer (ismir25-ai-music-detector)."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import interpolate

from .common import N_FFT, SR

FREQS = np.linspace(0, SR / 2, N_FFT // 2 + 1)
_WINDOW = np.hanning(N_FFT + 1)[:-1].astype(np.float64)  # Hann periódica, como torch.hann_window


def mean_db_spectrum(x):
    """STFT de potencia (n_fft=16384, hop=8192, centrada, padding reflect), en dB, promediada en el tiempo.

    Equivale a torchaudio.transforms.Spectrogram(n_fft=N_FFT, power=2) + 10*log10(clip(., 1e-10, 1e6)) + mean.
    """
    hop = N_FFT // 2
    xp = np.pad(x.astype(np.float64), N_FFT // 2, mode="reflect")
    frames = sliding_window_view(xp, N_FFT)[::hop] * _WINDOW
    p = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    db = 10 * np.log10(np.clip(p, 1e-10, 1e6))
    return db.mean(axis=0).astype(np.float32)


def lower_hull(c, area=10):
    """Mínimo de cada ventana deslizante de `area` bins (índices únicos, ordenados) + extremos."""
    idx = np.unique(np.argmin(sliding_window_view(c, area), axis=1) + np.arange(len(c) - area + 1))
    if idx[0] != 0:
        idx = np.r_[0, idx]
    if idx[-1] != len(c) - 1:
        idx = np.r_[idx, len(c) - 1]
    return idx, c[idx]


def band_mask(band):
    return (band[0] < FREQS) & (FREQS < band[1])


def fingerprint(mean_db, band=(1000, 8000), min_db=-45, max_db=5):
    """Picos sobre la envolvente inferior, recortados a [0, max_db] dB y normalizados por el máximo."""
    m = band_mask(band)
    f, c = FREQS[m], mean_db[m].astype(np.float64)
    i, h = lower_hull(c)
    hull = np.clip(interpolate.interp1d(f[i], h, kind="quadratic")(f), min_db, None)
    r = np.clip(c - hull, 0, max_db)
    return (r / (1e-6 + r.max())).astype(np.float32)


def fingerprints(spectra, band=(1000, 8000)):
    return np.stack([fingerprint(s, band) for s in spectra])
