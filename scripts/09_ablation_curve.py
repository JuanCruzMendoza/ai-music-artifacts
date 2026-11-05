"""Curva de ablación: cuánto pierde el modelo YA ENTRENADO a medida que se pisan más picos (umbral K más bajo),
comparado con pisar la misma cantidad de bins al azar. Solo dev, 5-fold CV agrupada.

Usa las mismas definiciones que 08_peaks.py (picos sobre la diferencia con las reales, calculados solo con
el entrenamiento de cada fold, ensanchados ±1 bin; reemplazo con los bins de una canción real al azar).

Salida: data/results/peaks_ablation_curve.csv
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common import DATA, SEED
from src.data import X_of, load_view

pk = importlib.import_module("08_peaks")
RES = DATA / "results"
KS = [10, 8, 6, 5, 4, 3, 2.5, 2]
N_RANDOM = 10


def main():
    choice = json.load(open(RES / "choice.json"))
    band, C = tuple(choice["band"]), choice["C"]
    m, spec = load_view("canon")
    X = X_of(spec, band).astype(np.float64)
    y, gen = m["label"].values, m["generator"].values
    fold = pd.read_csv(RES / "folds.csv").set_index("uid").loc[m["uid"], "fold"].values
    rng = np.random.default_rng(SEED)

    rows = []
    for k in range(5):
        tr, te = fold != k, fold == k
        clf = pk.l2(C).fit(X[tr], y[tr])
        real_tr = np.where(tr & (y == 0))[0]
        donors = X[rng.choice(real_tr, te.sum())]
        rows.append({"fold": k, "K": np.inf, "caso": "sin tocar", "n_bins": 0, **pk.metrics(clf.predict(X[te]), y[te], gen[te])})
        for K in KS:
            pk.K = K
            peaks = pk.peaks_of(X, gen, tr)
            P = pk.dilate(peaks["suno_v3.5"] | peaks["udio_130"])
            Xa = X[te].copy()
            Xa[:, P] = donors[:, P]
            rows.append({"fold": k, "K": K, "caso": "picos", "n_bins": int(P.sum()),
                         "n_suno": int(peaks["suno_v3.5"].sum()), "n_udio": int(peaks["udio_130"].sum()),
                         **pk.metrics(clf.predict(Xa), y[te], gen[te])})
            for r in range(N_RANDOM):
                R = pk.random_like(int(P.sum()), rng, P)
                Xr = X[te].copy()
                Xr[:, R] = donors[:, R]
                rows.append({"fold": k, "K": K, "caso": "azar", "rep": r, "n_bins": int(R.sum()),
                             **pk.metrics(clf.predict(Xr), y[te], gen[te])})
        print(f"fold {k} listo")

    df = pd.DataFrame(rows)
    df.to_csv(RES / "peaks_ablation_curve.csv", index=False)
    cols = ["n_bins", "real", "suno_v3.5", "udio_130", "balanced accuracy"]
    out = df.groupby(["K", "caso"])[cols].mean()
    out[cols[1:]] *= 100
    print(out.round(1).to_string())
    print(df[df.caso == "picos"].groupby("K")[["n_suno", "n_udio"]].mean().round(0).to_string())


if __name__ == "__main__":
    main()
