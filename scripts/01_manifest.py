"""Arma data/meta/manifest.csv: qué archivos van a dev / heldout y dónde están dentro de los zips remotos.

Solo selecciona archivos (metadata); no baja audio.
"""
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import remotezip
from src.common import FMA_META_ZIP, FMA_ZIP, GENERATORS, META, SEED, SEEN, SONICS_PARTS, UNSEEN

N_DEV_PER_GEN = 1800
N_HELDOUT_PER_GEN = 400


def fma_tracks():
    out = META / "fma_small_tracks.csv"
    if out.exists():
        return pd.read_csv(out, dtype={"track_id": int})
    z = zipfile.ZipFile(io.BufferedReader(remotezip.HttpFile(FMA_META_ZIP), buffer_size=1 << 22))
    t = pd.read_csv(z.open("fma_metadata/tracks.csv"), index_col=0, header=[0, 1], low_memory=False)
    t = t[t["set", "subset"] == "small"]
    df = pd.DataFrame({
        "track_id": t.index,
        "split": t["set", "split"].values,
        "artist_id": t["artist", "id"].values,
        "genre": t["track", "genre_top"].values,
        "bit_rate": t["track", "bit_rate"].values,
    })
    df.to_csv(out, index=False)
    return df


def sample_sonics_ids(df, n_files, rng):
    """Muestrea canciones (id) y toma sus dos versiones (_0, _1) hasta juntar n_files archivos."""
    ids = df["song_id"].unique()
    rng.shuffle(ids)
    counts = df.groupby("song_id").size()
    keep, total = [], 0
    for i in ids:
        if total >= n_files:
            break
        keep.append(i)
        total += counts[i]
    return df[df["song_id"].isin(keep)]


def main():
    rng = np.random.default_rng(SEED)

    # --- FMA (reales): split oficial, filtrado por artista
    t = fma_tracks()
    t = t[t["split"].isin(["training", "validation", "test"])]
    fma_idx = remotezip.index(FMA_ZIP)
    rows = []
    for r in t.itertuples():
        member = f"fma_small/{r.track_id // 1000:03d}/{r.track_id:06d}.mp3"
        off, size, ctype = fma_idx[member]
        rows.append(dict(
            uid=f"fma_{r.track_id:06d}", source="fma", label=0, generator="real",
            partition="dev" if r.split in ("training", "validation") else "heldout",
            group=f"artist_{r.artist_id}", genre=r.genre, src_bit_rate=r.bit_rate,
            zip_url=FMA_ZIP, member=member, offset=off, csize=size, ctype=ctype,
        ))
    fma = pd.DataFrame(rows)

    # --- SONICS (fakes)
    f = pd.read_csv(META / "fake_songs.csv", low_memory=False)
    f["generator"] = f["algorithm"].map(GENERATORS)
    f["song_id"] = f["filename"].str.extract(r"fake_(\d+)_")[0].astype(int)
    with ThreadPoolExecutor(10) as ex:
        idxs = list(ex.map(remotezip.index, SONICS_PARTS))
    where = {name.split("/")[-1][:-4]: (url, name, *v) for url, idx in zip(SONICS_PARTS, idxs) for name, v in idx.items()}

    parts = []
    for gen in SEEN:
        g = f[(f.generator == gen) & (f.split == "train")]
        parts.append(sample_sonics_ids(g, N_DEV_PER_GEN, rng).assign(partition="dev"))
    for gen in SEEN + UNSEEN:  # el split 'valid' de SONICS no se usa
        g = f[(f.generator == gen) & (f.split == "test")]
        parts.append(sample_sonics_ids(g, N_HELDOUT_PER_GEN, rng).assign(partition="heldout"))
    s = pd.concat(parts)
    s = s.assign(
        uid="sonics_" + s["filename"], source="sonics", label=1, group="song_" + s["song_id"].astype(str),
        genre=s["genre"], src_bit_rate=s["bit_rate"],
    )
    loc = pd.DataFrame([where[fn] for fn in s["filename"]], columns=["zip_url", "member", "offset", "csize", "ctype"])
    s = pd.concat([s.reset_index(drop=True), loc], axis=1)
    s["partition"] = np.where(s["partition"] == "dev", "dev", np.where(s["generator"].isin(SEEN), "heldout_seen", "heldout_unseen"))

    cols = list(fma.columns)
    m = pd.concat([fma, s[cols]], ignore_index=True)
    # las reales del heldout sirven para heldout_seen y heldout_unseen
    m.loc[m["partition"] == "heldout", "partition"] = "heldout_real"
    m.to_csv(META / "manifest.csv", index=False)
    print(m.groupby(["partition", "generator"]).size().to_string())


if __name__ == "__main__":
    main()
