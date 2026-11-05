"""E2 en dev: validación cruzada (5 folds agrupados), elección de banda y C, comparación de modelos y control de codec.

Salidas en data/results/: folds.csv, cv_*.csv, choice.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, SEED
from src.data import X_of, load_view

RES = DATA / "results"
# LAME -V 4 a 16 kHz corta en ~7,3 kHz (ver log): se prueban bandas que terminan antes de ese borde
BANDS = [(1000, 8000), (500, 7200), (1000, 7200), (2000, 7200), (5000, 7200)]
CS = [0.001, 0.003, 0.01, 0.03, 0.1, 1, 10]


def logreg(C=1.0):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000)


def make_folds(m):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    fold = np.empty(len(m), int)
    for k, (_, te) in enumerate(cv.split(m, m["label"], m["group"])):
        fold[te] = k
    return pd.DataFrame({"uid": m["uid"], "fold": fold})


def cv_eval(model_fn, X, m, folds):
    """Devuelve métricas por fold: accuracy por generador, accuracy balanceada y ROC-AUC."""
    f = folds.set_index("uid").loc[m["uid"], "fold"].values
    y = m["label"].values
    rows = []
    for k in range(5):
        tr, te = f != k, f == k
        clf = model_fn().fit(X[tr], y[tr])
        score = clf.decision_function(X[te])
        pred = (score > 0).astype(int)
        ok = pd.Series(pred == y[te]).groupby(m["generator"].values[te]).mean()
        acc_neg, acc_pos = (pred[y[te] == 0] == 0).mean(), (pred[y[te] == 1] == 1).mean()
        rows.append({"fold": k, **ok.to_dict(), "bal_acc": (acc_neg + acc_pos) / 2, "auc": roc_auc_score(y[te], score)})
    return pd.DataFrame(rows)


def summary(df):
    s = df.drop(columns="fold").agg(["mean", "std"]).T
    return s.apply(lambda r: f"{100 * r['mean']:.2f} ± {100 * r['std']:.2f}", axis=1)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    m, spec = load_view("canon")
    print(m.groupby("generator").size().to_string())
    folds = make_folds(m)
    folds.to_csv(RES / "folds.csv", index=False)

    # 1) Banda (LogReg con C=1)
    out = {}
    for band in BANDS:
        out[f"{band[0]}-{band[1]}"] = summary(cv_eval(logreg, X_of(spec, band), m, folds))
        print("banda", band, dict(out[f"{band[0]}-{band[1]}"]))
    pd.DataFrame(out).to_csv(RES / "cv_bands.csv")
    bal = {k: float(v["bal_acc"].split(" ")[0]) for k, v in out.items()}
    best_band = tuple(int(b) for b in max(bal, key=bal.get).split("-"))
    X = X_of(spec, best_band)

    # 2) C de la LogReg
    out = {}
    for C in CS:
        out[C] = summary(cv_eval(lambda: logreg(C), X, m, folds))
        print("C", C, dict(out[C]))
    pd.DataFrame(out).to_csv(RES / "cv_C.csv")
    balc = {C: float(v["bal_acc"].split(" ")[0]) for C, v in out.items()}
    best_C = max(balc, key=balc.get)

    # 3) Modelos
    models = {
        "LogReg": lambda: logreg(best_C),
        "SVM lineal": lambda: LinearSVC(C=0.1, class_weight="balanced", max_iter=20000),
        "SVM RBF": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
    }
    out = {}
    for name, fn in models.items():
        df = cv_eval(fn, X, m, folds)
        df.to_csv(RES / f"cv_model_{name.replace(' ', '_')}.csv", index=False)
        out[name] = summary(df)
        print(name, dict(out[name]))
    pd.DataFrame(out).to_csv(RES / "cv_models.csv")

    # 4) Control de codec: FMA sin re-encodear (como el paper)
    m2, spec2 = load_view("canon", fma_base="orig")
    PAPER = (1000, 8000)
    out_codec = {
        f"igualado, banda {best_band}": out["LogReg"],
        f"igualado, banda del paper {PAPER}": summary(cv_eval(lambda: logreg(best_C), X_of(spec, PAPER), m, folds)),
        f"sin igualar, banda {best_band}": summary(cv_eval(lambda: logreg(best_C), X_of(spec2, best_band), m2, folds)),
        f"sin igualar, banda del paper {PAPER}": summary(cv_eval(lambda: logreg(best_C), X_of(spec2, PAPER), m2, folds)),
    }
    pd.DataFrame(out_codec).to_csv(RES / "cv_codec.csv")
    for k, v in out_codec.items():
        print("codec:", k, dict(v))

    # 5) ¿La huella distingue el codec por sí solo? FMA re-encodeado (1) vs FMA original (0), solo reales
    sel_c = (m["label"] == 0).values
    sel_o = (m2["label"] == 0).values
    a = m[sel_c].assign(label=1, generator="fma_lame")
    b = m2[sel_o].assign(label=0, generator="real")
    a["uid"], b["uid"] = a["uid"] + "_lame", b["uid"] + "_orig"
    mm = pd.concat([a, b], ignore_index=True)
    XX = np.concatenate([X[sel_c], X_of(spec2[sel_o], best_band)])
    fold_of = folds.set_index("uid")["fold"]
    ff = pd.DataFrame({"uid": mm["uid"], "fold": fold_of.loc[mm["uid"].str[:-5]].values})
    df = cv_eval(lambda: logreg(best_C), XX, mm, ff)
    print("FMA LAME vs FMA original:", dict(summary(df)))
    summary(df).to_csv(RES / "cv_codec_only.csv")

    json.dump({"band": best_band, "C": best_C}, open(RES / "choice.json", "w"))
    print("elegido:", best_band, best_C)


if __name__ == "__main__":
    main()
