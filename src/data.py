"""Carga de espectros + metadata y armado de la matriz de huellas."""
import numpy as np
import pandas as pd

from .common import FEAT, META, split_suffix
from .fingerprint import fingerprints


def load_view(base="canon", transform="none", fma_base=None, allow_heldout=False):
    """Devuelve (meta, spectra) alineados. `fma_base` permite tomar FMA de otra vista (p. ej. 'orig')."""
    def _load(b, t):
        z = np.load(FEAT / f"{b}__{t}{split_suffix(allow_heldout)}.npz")
        return pd.Index(z["uids"]), z["spectra"]

    uids, spec = _load(base, transform)
    if fma_base is not None:
        keep = ~uids.str.startswith("fma_")
        u2, s2 = _load(fma_base, transform)
        uids, spec = uids[keep].append(u2), np.concatenate([spec[keep], s2])
    m = pd.read_csv(META / "manifest.csv").set_index("uid").loc[uids].reset_index()
    is_dev = m["partition"] == "dev"
    assert (~is_dev).all() if allow_heldout else is_dev.all(), "dev y heldout no se mezclan"
    return m, spec


def X_of(spec, band):
    return fingerprints(spec, band)
