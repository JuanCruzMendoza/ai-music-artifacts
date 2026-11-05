"""Transformaciones de robustez (E3/E4). Se aplican al audio canónico ya recortado (16 kHz, mono)."""
import zlib

import librosa
import numpy as np
import soxr
from scipy import signal

from .audio import load_mono, mp3_encode
from .common import SR


def pitch(x, steps):
    return librosa.effects.pitch_shift(x, sr=SR, n_steps=steps)


def speed(x, pct):
    """Cambio de velocidad por resampleo: todo suena pct% más rápido y las frecuencias se multiplican por (1+pct/100)."""
    f = 1 + pct / 100
    return soxr.resample(x, SR * f, SR).astype(np.float32)


def mp3(x, kbps):
    return load_mono(mp3_encode(x, args=("-b:a", f"{kbps}k")))


def lowpass(x, fc):
    sos = signal.butter(8, fc, btype="low", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def noise(x, snr_db, key):
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    p = np.mean(x.astype(np.float64) ** 2) / 10 ** (snr_db / 10)
    return (x + rng.normal(0, np.sqrt(p), len(x))).astype(np.float32)


TRANSFORMS = {
    "pitch-2": lambda x, k: pitch(x, -2),
    "pitch-1": lambda x, k: pitch(x, -1),
    "pitch+1": lambda x, k: pitch(x, 1),
    "pitch+2": lambda x, k: pitch(x, 2),
    "speed-5": lambda x, k: speed(x, -5),
    "speed-2": lambda x, k: speed(x, -2),
    "speed+2": lambda x, k: speed(x, 2),
    "speed+5": lambda x, k: speed(x, 5),
    "mp3_128": lambda x, k: mp3(x, 128),
    "mp3_64": lambda x, k: mp3(x, 64),
    "lp_6k": lambda x, k: lowpass(x, 6000),
    "lp_4k": lambda x, k: lowpass(x, 4000),
    "noise_30": lambda x, k: noise(x, 30, k),
    "noise_20": lambda x, k: noise(x, 20, k),
}

# Magnitudes que el augmentation NO vio: solo se usan para evaluar (control de E4)
UNSEEN_TRANSFORMS = {
    "pitch+0.5": lambda x, k: pitch(x, 0.5),
    "speed+3": lambda x, k: speed(x, 3),
}
ALL_TRANSFORMS = {**TRANSFORMS, **UNSEEN_TRANSFORMS}
