"""Baja los MP3 de dev (FMA y SONICS) leyendo solo esos archivos de los zips remotos.

Uso:
  python scripts/02_download.py --per-gen 20   # subconjunto chico (20 por generador)
  python scripts/02_download.py                # todo dev
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import remotezip
from src.common import META, SEED, raw_path, require_dev


def subset(m, per_gen):
    if per_gen is None:
        return m
    return m.groupby("generator", group_keys=False).apply(lambda g: g.sample(min(per_gen, len(g)), random_state=SEED))


def download(row):
    out = raw_path(row)
    if out.exists():
        return 0
    data = remotezip.fetch(row["zip_url"], int(row["offset"]), int(row["csize"]), int(row["ctype"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(out)
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-gen", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--allow-heldout", action="store_true", help="requiere autorización explícita")
    a = ap.parse_args()

    m = require_dev(pd.read_csv(META / "manifest.csv"), a.allow_heldout)
    m = subset(m, a.per_gen)
    rows = [r for _, r in m.iterrows()]
    total, failed = 0, []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(download, r): r["uid"] for r in rows}
        for fu in tqdm(as_completed(futs), total=len(futs), mininterval=10):
            try:
                total += fu.result()
            except Exception as e:  # noqa: BLE001
                failed.append((futs[fu], repr(e)))
    print(f"bajados {total / 1e9:.2f} GB; fallidos: {len(failed)}")
    for f in failed[:10]:
        print(f)


if __name__ == "__main__":
    main()
