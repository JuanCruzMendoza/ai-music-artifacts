"""Lectura de audio y re-encodeo MP3 (LAME vía ffmpeg)."""
import io
import os
import subprocess
import tempfile

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import soxr

from .common import SR

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def load_mono(path_or_bytes, sr=SR):
    """Decodifica (MP3/WAV), pasa a mono y resamplea a `sr` con soxr. Devuelve float32 en [-1, 1]."""
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    x, fs = sf.read(src, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != sr:
        x = soxr.resample(x, fs, sr).astype(np.float32)
    return x


def mp3_encode(x, sr=SR, args=("-q:a", "4")):
    """Encodea con libmp3lame. Por defecto `-q:a 4`, que equivale a LAME `-V 4` (lo que usa SONICS)."""
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    # Se escribe a un archivo (no a un pipe) para que ffmpeg complete el header Xing/LAME al final;
    # con pipe el header queda con un conteo de frames incorrecto y el decoder corta el audio.
    fd, tmp = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac", "1",
               "-i", "pipe:0", "-map_metadata", "-1", "-c:a", "libmp3lame", *args, tmp]
        subprocess.run(cmd, input=pcm, capture_output=True, check=True)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        os.remove(tmp)


def crop(x, seconds, sr=SR):
    n = int(round(seconds * sr))
    return x[:n] if len(x) >= n else None
