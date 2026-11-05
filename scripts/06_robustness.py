"""E3 (robustez) y E4 (augmentation) dentro de dev.

El heldout no se toca: se usa el fold 0 de dev como evaluación y los folds 1-4 como entrenamiento.

  python scripts/06_robustness.py prepare   # escribe las listas de uids a transformar
  (calcular espectros con 03_spectra.py --transform T --uids data/results/robust_uids_T.csv)
  python scripts/06_robustness.py eval
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, SEED
from src.data import X_of, load_view
from src.transforms import TRANSFORMS, UNSEEN_TRANSFORMS

RES = DATA / "results"
EVAL_FOLD = 0


def split():
    folds = pd.read_csv(RES / "folds.csv").set_index("uid")["fold"]
    return folds[folds == EVAL_FOLD].index, folds[folds != EVAL_FOLD].index


def prepare():
    ev, tr = split()
    rng = np.random.default_rng(SEED)
    names = list(TRANSFORMS)
    aug = pd.DataFrame({"uid": tr, "transform": rng.choice(names, len(tr))})
    aug.to_csv(RES / "augment_assignment.csv", index=False)
    for t in names:
        uids = pd.Index(ev).append(pd.Index(aug.loc[aug["transform"] == t, "uid"]))
        pd.DataFrame({"uid": uids}).to_csv(RES / f"robust_uids_{t}.csv", index=False)
    for t in UNSEEN_TRANSFORMS:  # solo evaluación
        pd.DataFrame({"uid": ev}).to_csv(RES / f"robust_uids_{t}.csv", index=False)
    print(f"eval: {len(ev)} archivos x {len(names)} transformaciones; augmentation: {len(tr)} archivos")


def per_class(clf, X, m):
    pred = clf.predict(X)
    ok = pd.Series(pred == m["label"].values).groupby(m["generator"].values).mean()
    ok["IA (total)"] = (pred[m["label"].values == 1] == 1).mean()
    ok["balanced accuracy"] = (ok["real"] + ok["IA (total)"]) / 2
    return ok


def fit_models(band, C):
    """Entrena en los folds 1-4 de dev: modelo original y modelo con augmentation."""
    _, tr = split()
    m, spec = load_view("canon")
    m = m.set_index("uid")
    X = pd.DataFrame(X_of(spec, band), index=m.index)

    def fit(Xtr, ytr):
        return LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Xtr, ytr)

    clean = fit(X.loc[tr].values, m.loc[tr, "label"].values)
    # augmentation: cada archivo de entrenamiento suma una copia con una transformación al azar
    aug = pd.read_csv(RES / "augment_assignment.csv")
    Xa, ya = [X.loc[tr].values], [m.loc[tr, "label"].values]
    for t in TRANSFORMS:
        mt, st = load_view("canon", t)
        idx = pd.Index(mt["uid"])
        u = aug.loc[aug["transform"] == t, "uid"]
        u = u[u.isin(idx)]
        Xa.append(X_of(st[idx.get_indexer(u)], band))
        ya.append(m.loc[u, "label"].values)
    return clean, fit(np.concatenate(Xa), np.concatenate(ya))


def evaluate():
    choice = json.load(open(RES / "choice.json"))
    band, C = tuple(choice["band"]), choice["C"]
    ev, _ = split()
    clean, augmented = fit_models(band, C)
    m, spec = load_view("canon")
    m = m.set_index("uid")
    X = pd.DataFrame(X_of(spec, band), index=m.index)
    views = {}
    for t in {**TRANSFORMS, **UNSEEN_TRANSFORMS}:
        mt, st = load_view("canon", t)
        mt = mt.set_index("uid")
        sel = mt.index.isin(ev)
        views[t] = (mt[sel], X_of(st[sel], band))

    rows = []
    for name, clf in [("original", clean), ("con augmentation", augmented)]:
        rows.append({"modelo": name, "transformación": "ninguna", **per_class(clf, X.loc[ev].values, m.loc[ev]).to_dict()})
        for t, (mt, Xt) in views.items():
            rows.append({"modelo": name, "transformación": t, **per_class(clf, Xt, mt).to_dict()})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "robustness.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    {"prepare": prepare, "eval": evaluate}[sys.argv[1]]()
