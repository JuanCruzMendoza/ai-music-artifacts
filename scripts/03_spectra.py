"""Calcula el espectro promedio en dB (0-8 kHz) de cada archivo de dev ya bajado.

Vistas (--base):
  canon : FMA re-encodeado como SONICS (16 kHz mono, LAME -V 4); SONICS tal cual.   <- pipeline principal
  orig  : FMA solo pasado a 16 kHz mono, sin re-encodear (como el paper).           <- control de codec
Con --transform se aplica una transformación de robustez (src/transforms.py) al audio ya recortado.

Salida: data/features/{base}__{transform|none}.npz  (uids, spectra)
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio import crop, load_mono, mp3_encode
from src.common import CANON, CROP_S, FEAT, META, raw_path, require_dev, split_suffix
from src.fingerprint import mean_db_spectrum
from src.transforms import ALL_TRANSFORMS as TRANSFORMS


def canonical_audio(row):
    if row["source"] == "sonics":
        return load_mono(str(raw_path(row)))
    out = CANON / "fma" / f"{row['uid']}.mp3"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.part")
        tmp.write_bytes(mp3_encode(load_mono(str(raw_path(row)))))
        tmp.rename(out)
    return load_mono(str(out))


def work(args):
    row, base, transform = args
    try:
        x = canonical_audio(row) if base == "canon" else load_mono(str(raw_path(row)))
        x = crop(x, CROP_S)
        if x is None:
            return row["uid"], None, "corto"
        if transform != "none":
            x = TRANSFORMS[transform](x, row["uid"])
        return row["uid"], mean_db_spectrum(x), None
    except Exception as e:  # noqa: BLE001
        return row["uid"], None, repr(e)


def out_path(base, transform, heldout=False):
    return FEAT / f"{base}__{transform}{split_suffix(heldout)}.npz"


def load(base, transform="none", heldout=False):
    z = np.load(out_path(base, transform, heldout))
    return pd.Index(z["uids"]), z["spectra"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["canon", "orig"], default="canon")
    ap.add_argument("--transform", default="none")
    ap.add_argument("--uids", help="CSV con columna uid para restringir los archivos")
    ap.add_argument("--source", choices=["fma", "sonics"], help="restringir a una fuente")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--allow-heldout", action="store_true", help="requiere autorización explícita")
    a = ap.parse_args()

    m = require_dev(pd.read_csv(META / "manifest.csv"), a.allow_heldout)
    m = m[[raw_path(r).exists() for _, r in m.iterrows()]]
    if a.uids:
        m = m[m["uid"].isin(pd.read_csv(a.uids)["uid"])]
    if a.source:
        m = m[m["source"] == a.source]

    out = out_path(a.base, a.transform, a.allow_heldout)
    done_uids, done_spec = (load(a.base, a.transform, a.allow_heldout) if out.exists() else (pd.Index([]), np.zeros((0, 8193), np.float32)))
    todo = m[~m["uid"].isin(done_uids)]
    print(f"{out.name}: {len(done_uids)} hechos, {len(todo)} por calcular")
    if len(todo) == 0:
        return

    uids, specs, errors = [], [], []
    with Pool(a.workers) as p:
        jobs = [(r, a.base, a.transform) for _, r in todo.iterrows()]
        for uid, s, err in tqdm(p.imap_unordered(work, jobs, chunksize=4), total=len(jobs), mininterval=15):
            if err:
                errors.append((uid, err))
            else:
                uids.append(uid)
                specs.append(s)
    FEAT.mkdir(parents=True, exist_ok=True)
    all_uids = np.r_[done_uids.values, np.array(uids, dtype=object)].astype(str)
    all_spec = np.concatenate([done_spec, np.stack(specs)]) if specs else done_spec
    np.savez(out, uids=all_uids, spectra=all_spec)
    print(f"guardados {len(all_uids)}; errores: {len(errors)}")
    for e in errors[:10]:
        print("  ", e)
    if errors:
        pd.DataFrame(errors, columns=["uid", "error"]).to_csv(out.with_suffix(".errors.csv"), index=False)


if __name__ == "__main__":
    main()
