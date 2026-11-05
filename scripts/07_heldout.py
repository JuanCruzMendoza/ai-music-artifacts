"""Evaluación final sobre el heldout. NO CORRER sin autorización explícita.

Pasos (todos con --allow-heldout):
  python scripts/02_download.py --allow-heldout
  python scripts/03_spectra.py --allow-heldout                               # codec igualado
  python scripts/03_spectra.py --allow-heldout --base orig --source fma      # control de codec
  for t in <transformaciones>: python scripts/03_spectra.py --allow-heldout --transform $t
  python scripts/07_heldout.py --allow-heldout

Pipeline congelado en dev: banda y C de data/results/choice.json.
  - E2: LogReg (principal) y SVM RBF entrenados en TODO dev; se evalúan en heldout_seen y heldout_unseen.
  - Control de codec: lo mismo con FMA sin re-encodear y la banda del paper.
  - E3/E4: los mismos dos modelos de 06_robustness.py (entrenados en folds 1-4 de dev, sin y con augmentation).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, FEAT
from src.data import X_of, load_view
from src.transforms import ALL_TRANSFORMS

RES = DATA / "results"


def report(clf, X, m):
    pred = clf.predict(X)
    ok = pd.Series(pred == m["label"].values).groupby(m["generator"].values).mean() * 100
    return ok.round(2).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-heldout", action="store_true", required=True)
    ap.parse_args()
    c = json.load(open(RES / "choice.json"))
    band, C = tuple(c["band"]), c["C"]
    paper = (1000, 8000)

    md, sd = load_view("canon")
    mh, sh = load_view("canon", allow_heldout=True)
    out = {}
    models = {
        "LogReg": lambda: LogisticRegression(C=C, class_weight="balanced", max_iter=5000),
        "SVM RBF": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
    }
    for name, fn in models.items():
        clf = fn().fit(X_of(sd, band), md["label"].values)
        out[f"{name}, codec igualado"] = report(clf, X_of(sh, band), mh)

    mdo, sdo = load_view("canon", fma_base="orig")
    mho, sho = load_view("canon", fma_base="orig", allow_heldout=True)
    clf = models["LogReg"]().fit(X_of(sdo, paper), mdo["label"].values)
    out["LogReg, sin igualar, banda del paper"] = report(clf, X_of(sho, paper), mho)
    pd.DataFrame(out).to_csv(RES / "heldout_e2.csv")
    print(pd.DataFrame(out).to_string())

    # E3/E4: reentrena los modelos de 06 (folds 1-4) y los evalúa en el heldout transformado
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib
    rob = importlib.import_module("06_robustness")
    clean, augmented = rob.fit_models(band, C)
    rows = []
    for t in ["none", *ALL_TRANSFORMS]:
        if not (FEAT / f"canon__{t}__heldout.npz").exists():
            continue
        mt, st = load_view("canon", t, allow_heldout=True)
        Xt = X_of(st, band)
        for name, clf in [("original", clean), ("con augmentation", augmented)]:
            rows.append({"modelo": name, "transformación": t, **report(clf, Xt, mt)})
    pd.DataFrame(rows).to_csv(RES / "heldout_robustness.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
