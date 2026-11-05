"""¿El modelo usa los picos que vemos? (solo dev, 5-fold CV agrupada)

Paso 0. Picos: bins donde la huella promedio de un generador (Suno, Udio) supera a la de las reales
        por más de K desvíos robustos respecto de su entorno (~50 Hz). Se calculan SOLO con los folds de
        entrenamiento y se ensanchan ±1 bin.
Paso 1. Concentración: qué fracción del peso del modelo cae en los picos vs. la fracción de bins que son pico.
Paso 2. Ablación: en el fold de test se reemplazan los bins de pico por los de una canción real de
        entrenamiento elegida al azar; control: misma cantidad de bins al azar.
Paso 3. Reentrenar solo con picos / sin picos / solo con bins al azar (misma cantidad).
Lasso.  Regresión logística L1 con grilla de lambda (= 1/C), elegida por balanced accuracy en CV;
        se compara qué bins selecciona con los picos.

Salidas: data/results/peaks_*.csv, peaks_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, median_filter
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, SEED
from src.data import X_of, load_view
from src.fingerprint import FREQS, band_mask

RES = DATA / "results"
K = 5            # umbral en desvíos robustos
WIN = 51         # ventana del entorno local (~50 Hz)
N_RANDOM = 20    # repeticiones del control al azar
LASSO_C = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]  # lambda = 1/C


def peaks_of(X, gen, train):
    """Bins de pico por generador (dict) usando solo las canciones de `train`."""
    real = X[train & (gen == "real")].mean(0)
    out = {}
    for g in ["suno_v3.5", "udio_130"]:
        d = X[train & (gen == g)].mean(0) - real
        base = median_filter(d, WIN, mode="nearest")
        mad = median_filter(np.abs(d - base), WIN, mode="nearest") * 1.4826 + 1e-9
        out[g] = (d - base) / mad > K
    return out


def dilate(mask):
    return binary_dilation(mask, iterations=1)


def random_like(n_bins, rng, exclude):
    """Exactamente `n_bins` bins al azar, en bloques de 3 contiguos (como un pico ensanchado), fuera de `exclude`."""
    r = np.zeros_like(exclude)
    blocked = dilate(exclude)
    while r.sum() < n_bins:
        i = rng.integers(1, len(r) - 1)
        if not blocked[i - 1:i + 2].any() and not r[i - 1:i + 2].any():
            r[i - 1:i + 2] = True
    extra = int(r.sum() - n_bins)  # recorta el último bloque si se pasó
    if extra:
        r[np.where(r)[0][-extra:]] = False
    return r


def metrics(pred, y, gen):
    ok = pred == y
    real, ia = ok[y == 0].mean(), ok[y == 1].mean()
    return {"real": real, "suno_v3.5": ok[gen == "suno_v3.5"].mean(), "udio_130": ok[gen == "udio_130"].mean(),
            "balanced accuracy": (real + ia) / 2}


def l2(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000)


def l1(C):
    return LogisticRegression(l1_ratio=1, solver="liblinear", C=C, class_weight="balanced", max_iter=2000)


def concentration(w, peak):
    """Fracción del peso positivo (que empuja a 'IA') en los picos, y cuántas veces más que el azar."""
    pos = np.clip(w, 0, None)
    share = pos[peak].sum() / pos.sum()
    return share, share / peak.mean()


def main():
    choice = json.load(open(RES / "choice.json"))
    band, C = tuple(choice["band"]), choice["C"]
    m, spec = load_view("canon")
    X = X_of(spec, band).astype(np.float64)
    f = FREQS[band_mask(band)]
    y, gen = m["label"].values, m["generator"].values
    fold = pd.read_csv(RES / "folds.csv").set_index("uid").loc[m["uid"], "fold"].values
    rng = np.random.default_rng(SEED)

    rows_abl, rows_ret, rows_las = [], [], []
    for k in range(5):
        tr, te = fold != k, fold == k
        pk = peaks_of(X, gen, tr)
        P = dilate(pk["suno_v3.5"] | pk["udio_130"])
        print(f"fold {k}: picos Suno {pk['suno_v3.5'].sum()}, Udio {pk['udio_130'].sum()}, bins de pico (±1) {P.sum()}")
        # reemplazo: los valores de una canción real de entrenamiento elegida al azar (conserva la distribución real)
        donors = X[tr & (y == 0)][rng.integers(0, (tr & (y == 0)).sum(), te.sum())]

        # --- Paso 2: ablación sobre el modelo completo
        clf = l2(C).fit(X[tr], y[tr])
        rows_abl.append({"fold": k, "caso": "sin tocar", **metrics(clf.predict(X[te]), y[te], gen[te])})
        Xa = X[te].copy()
        Xa[:, P] = donors[:, P]
        rows_abl.append({"fold": k, "caso": "picos borrados", **metrics(clf.predict(Xa), y[te], gen[te])})
        for r in range(N_RANDOM):
            R = random_like(int(P.sum()), rng, P)
            Xr = X[te].copy()
            Xr[:, R] = donors[:, R]
            rows_abl.append({"fold": k, "caso": "bins al azar borrados", "rep": r, **metrics(clf.predict(Xr), y[te], gen[te])})

        # --- Paso 3: reentrenar con subconjuntos de bins
        for name, cols in [("todos los bins", np.ones_like(P)), ("solo picos", P), ("sin picos", ~P)]:
            c = l2(C).fit(X[tr][:, cols], y[tr])
            rows_ret.append({"fold": k, "caso": name, "n_bins": int(cols.sum()), **metrics(c.predict(X[te][:, cols]), y[te], gen[te])})
        for r in range(5):
            R = random_like(int(P.sum()), rng, P)
            c = l2(C).fit(X[tr][:, R], y[tr])
            rows_ret.append({"fold": k, "caso": "solo bins al azar", "rep": r, "n_bins": int(R.sum()),
                             **metrics(c.predict(X[te][:, R]), y[te], gen[te])})

        # --- Lasso: grilla de lambda
        for Cl in LASSO_C:
            c = l1(Cl).fit(X[tr], y[tr])
            rows_las.append({"fold": k, "C": Cl, "lambda": 1 / Cl, "n_no_cero": int((c.coef_[0] != 0).sum()),
                             **metrics(c.predict(X[te]), y[te], gen[te])})
            print(f"  lasso C={Cl}: {rows_las[-1]['balanced accuracy']:.4f}, no-cero {rows_las[-1]['n_no_cero']}")

    abl, ret, las = pd.DataFrame(rows_abl), pd.DataFrame(rows_ret), pd.DataFrame(rows_las)
    abl.to_csv(RES / "peaks_ablation.csv", index=False)
    ret.to_csv(RES / "peaks_retrain.csv", index=False)
    las.to_csv(RES / "peaks_lasso_cv.csv", index=False)
    cols = ["real", "suno_v3.5", "udio_130", "balanced accuracy"]
    print("\nABLACIÓN\n", (abl.groupby("caso")[cols].mean() * 100).round(1))
    print("\nREENTRENAR\n", (ret.groupby("caso")[cols + ["n_bins"]].mean()).round(3))
    lcv = las.groupby("C")[cols + ["n_no_cero"]].mean()
    print("\nLASSO CV\n", lcv.round(3))
    best_C = float(lcv["balanced accuracy"].idxmax())

    # --- Paso 1 y Lasso final: modelos entrenados en todo dev, picos de todo dev
    allm = np.ones(len(y), bool)
    pk = peaks_of(X, gen, allm)
    P = dilate(pk["suno_v3.5"] | pk["udio_130"])
    w2 = l2(C).fit(X, y).coef_[0]
    w1 = l1(best_C).fit(X, y).coef_[0]
    nz = w1 != 0
    share2, enr2 = concentration(w2, P)
    share1, enr1 = concentration(w1, P)
    summary = {
        "K": K, "bins_banda": int(len(f)), "picos_suno": int(pk["suno_v3.5"].sum()), "picos_udio": int(pk["udio_130"].sum()),
        "bins_pico_dilatados": int(P.sum()), "fraccion_bins_pico": float(P.mean()),
        "picos_suno_multiplo_25Hz": float(np.mean(np.abs(f[pk["suno_v3.5"]] / 25 - np.round(f[pk["suno_v3.5"]] / 25)) * 25 < 1.5)),
        "L2_peso_positivo_en_picos": float(share2), "L2_enriquecimiento": float(enr2),
        "lasso_C": best_C, "lasso_lambda": 1 / best_C, "lasso_no_cero": int(nz.sum()),
        "lasso_no_cero_positivos": int((w1 > 0).sum()),
        "lasso_peso_positivo_en_picos": float(share1), "lasso_enriquecimiento": float(enr1),
        "lasso_precision_positivos": float(P[w1 > 0].mean()),  # de los bins que empujan a IA, % que son pico
        "lasso_recall_picos_suno": float((w1 > 0)[dilate(pk["suno_v3.5"])].reshape(-1).mean()),
    }
    # recall por pico (centro): ¿el lasso eligió algún bin dentro de ±1 de cada pico de Suno?
    centers = np.where(pk["suno_v3.5"])[0]
    summary["lasso_picos_suno_con_algun_bin"] = float(np.mean([(w1[max(0, i - 1):i + 2] > 0).any() for i in centers]))
    json.dump(summary, open(RES / "peaks_summary.json", "w"), indent=1)
    np.savez(RES / "peaks_final.npz", f=f, P=P, suno=pk["suno_v3.5"], udio=pk["udio_130"], w2=w2, w1=w1,
             mean_suno=X[gen == "suno_v3.5"].mean(0), mean_udio=X[gen == "udio_130"].mean(0), mean_real=X[gen == "real"].mean(0))
    print("\nRESUMEN", json.dumps(summary, indent=1))

    # --- Camino del Lasso en todo dev: para cada lambda, cuántos bins "IA" usa y qué % son picos
    path = []
    for Cl in LASSO_C:
        w = l1(Cl).fit(X, y).coef_[0]
        pos = w > 0
        if Cl == 0.01:
            np.save(RES / "lasso_w_C0.01.npy", w)  # lasso muy esparso (lambda = 100), para la figura 7
        cvr = lcv.loc[Cl]
        path.append({"lambda": 1 / Cl, "bal_acc_cv": 100 * cvr["balanced accuracy"], "suno_cv": 100 * cvr["suno_v3.5"],
                     "udio_cv": 100 * cvr["udio_130"], "bins_IA": int(pos.sum()),
                     "pct_bins_IA_pico": 100 * P[pos].mean() if pos.any() else np.nan})
    path = pd.DataFrame(path)
    path.to_csv(RES / "peaks_lasso_path.csv", index=False)
    print("\nCAMINO DEL LASSO\n", path.round(1).to_string())


if __name__ == "__main__":
    main()
