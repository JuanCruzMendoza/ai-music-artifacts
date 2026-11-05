from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
META = DATA / "meta"
RAW = DATA / "raw"          # MP3 originales (FMA y SONICS)
CANON = DATA / "canonical"  # FMA re-encodeado como SONICS (16 kHz mono, LAME -V 4)
FEAT = DATA / "features"    # espectros promedio en dB
FIGS = ROOT / "docs" / "figs"

SR = 16_000        # frecuencia de muestreo común
CROP_S = 29.0      # segundos analizados por archivo (ver docs/log.md)
N_FFT = 1 << 14    # como Deezer: 16384 -> ~0,98 Hz por bin a 16 kHz
SEED = 123

FMA_ZIP = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_META_ZIP = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
SONICS_HF = "https://huggingface.co/datasets/awsaf49/sonics/resolve/main"
SONICS_PARTS = [f"{SONICS_HF}/fake_songs/part_{i:02d}.zip" for i in range(1, 11)]

GENERATORS = {
    "chirp-v3.5": "suno_v3.5",
    "udio-120s": "udio_130",
    "chirp-v2-xxl-alpha": "suno_v2",
    "chirp-v3": "suno_v3",
    "udio-30s": "udio_32",
}
SEEN = ["suno_v3.5", "udio_130"]
UNSEEN = ["suno_v2", "suno_v3", "udio_32"]


def raw_path(row):
    return RAW / row["source"] / f"{row['uid']}.mp3"


def require_dev(df, allow_heldout=False):
    """Barrera: sin autorización explícita solo se procesa dev; con `allow_heldout` se procesa SOLO el heldout."""
    if allow_heldout:
        return df[df["partition"].str.startswith("heldout")]
    return df[df["partition"] == "dev"]


def split_suffix(allow_heldout):
    return "__heldout" if allow_heldout else ""
