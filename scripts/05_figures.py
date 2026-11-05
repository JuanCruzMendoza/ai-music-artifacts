"""Gráficos para docs/resultados.md (solo dev)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, FIGS
from src.data import X_of, load_view
from src.fingerprint import FREQS, band_mask
from src.plotstyle import COLORS, LABELS, TEXT2, plt

RES = DATA / "results"
GENS = ["real", "suno_v3.5", "udio_130"]


def choice():
    p = RES / "choice.json"
    if p.exists():
        c = json.load(open(p))
        return tuple(c["band"]), c["C"]
    return (1000, 8000), 1.0


def fig_fingerprints(m, X, band):
    """E1: huella promedio de cada clase."""
    f = FREQS[band_mask(band)] / 1000
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True, sharey=True)
    for ax, g in zip(axes, GENS):
        sel = (m["generator"] == g).values
        ax.plot(f, X[sel].mean(0), color=COLORS[g], lw=0.7)
        ax.set_title(f"{LABELS[g]}  (n = {sel.sum()})", fontsize=10)
        ax.set_ylabel("altura del pico\n(normalizada)")
    axes[-1].set_xlabel("frecuencia (kHz)")
    fig.suptitle("Huella espectral promedio: picos que sobresalen del espectro", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_huellas.png")
    plt.close(fig)


def fig_zoom(m, X, band, lo=2.9, hi=3.1):
    """Zoom en una ventana chica de frecuencias para ver los picos finos."""
    f = FREQS[band_mask(band)] / 1000
    w = (f >= lo) & (f <= hi)
    fig, ax = plt.subplots(figsize=(9, 3))
    for g in GENS:
        sel = (m["generator"] == g).values
        ax.plot(f[w], X[sel][:, w].mean(0), color=COLORS[g], label=LABELS[g], lw=1.2)
    ax.set_xlabel("frecuencia (kHz)")
    ax.set_ylabel("altura del pico")
    ax.legend(loc="upper right")
    ax.set_title(f"Zoom {lo}–{hi} kHz")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1b_zoom.png")
    plt.close(fig)


def fig_weights(m, X, band, C):
    """Pesos de la regresión logística: positivos = 'esto indica IA'."""
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(X, m["label"].values)
    f = FREQS[band_mask(band)] / 1000
    w = clf.coef_[0]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(f, 0, np.clip(w, 0, None), color="#eb6834", lw=0, label="indica IA")
    ax.fill_between(f, 0, np.clip(w, None, 0), color="#2a78d6", lw=0, label="indica real")
    ax.axhline(0, color=TEXT2, lw=0.6)
    ax.set_xlabel("frecuencia (kHz)")
    ax.set_ylabel("peso")
    ax.legend(loc="upper right", ncols=2)
    ax.set_title("Qué frecuencias mira el clasificador (regresión logística entrenada en todo dev)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_pesos.png")
    plt.close(fig)
    return f, w


def fig_codec(m, spec):
    """Control de codec: espectro promedio (dB) de FMA original vs FMA re-encodeado vs SONICS."""
    _, spec_o = load_view("canon", fma_base="orig")
    m_o, _ = load_view("canon", fma_base="orig")
    f = FREQS / 1000
    curves = [
        ("Real: FMA solo a 16 kHz (como el paper)", spec_o[(m_o["label"] == 0).values].mean(0), "#2a78d6", "--"),
        ("Real: FMA re-encodeado como SONICS (nuestro)", spec[(m["label"] == 0).values].mean(0), "#2a78d6", "-"),
        ("IA: SONICS (Suno + Udio)", spec[(m["label"] == 1).values].mean(0), "#eb6834", "-"),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for lab, c, col, ls in curves:
        ax.plot(f, c, color=col, ls=ls, label=lab, lw=1.4)
    ax.set_xlim(5, 8)
    w = f >= 5
    ax.set_ylim(-80, max(c[w].max() for _, c, _, _ in curves) + 5)
    ax.set_xlabel("frecuencia (kHz)")
    ax.set_ylabel("energía promedio (dB)")
    ax.legend(loc="lower left")
    ax.set_title("El MP3 de SONICS corta todo arriba de ~7,3 kHz; FMA original no")
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_codec.png")
    plt.close(fig)


def _mean(s):
    return float(str(s).split(" ")[0])


def fig_accuracy():
    """E2: accuracy por clase. Paper (Tabla 2) vs réplica sin igualar codec vs codec igualado."""
    cv = pd.read_csv(RES / "cv_codec.csv", index_col=0)
    col_ours = [c for c in cv.columns if c.startswith("igualado, banda (") and "paper" not in c][0]
    col_rep = [c for c in cv.columns if c.startswith("sin igualar, banda del paper")][0]
    conds = [
        ("Paper (Tabla 2)", {"real": 99.97, "suno_v3.5": 100.0, "udio_130": 100.0}, "#b9b8b3"),
        ("Réplica, como el paper (codec distinto)", {g: _mean(cv.loc[g, col_rep]) for g in GENS}, "#6250d6"),
        ("Codec igualado (nuestro)", {g: _mean(cv.loc[g, col_ours]) for g in GENS}, "#2a78d6"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    h = 0.26
    for i, (lab, vals, col) in enumerate(conds):
        y = np.arange(len(GENS)) + (i - 1) * h
        v = [vals[g] for g in GENS]
        ax.barh(y, v, height=h - 0.03, color=col, label=lab)
        for yy, vv in zip(y, v):
            ax.text(vv + 0.15, yy, f"{vv:.2f}".rstrip("0").rstrip("."), va="center", fontsize=8, color=TEXT2)
    ax.set_yticks(np.arange(len(GENS)), [LABELS[g] for g in GENS])
    ax.invert_yaxis()
    ax.set_xlim(80, 101.5)
    ax.set_xlabel("% de canciones bien clasificadas (validación cruzada en dev)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncols=3, fontsize=8.5)
    ax.set_title("Igualar el MP3 baja el acierto, sobre todo en Udio")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_accuracy.png")
    plt.close(fig)


def fig_robustness():
    """E3/E4: acierto en reales y en IA bajo cada transformación (fold 0 de dev)."""
    p = RES / "robustness.csv"
    if not p.exists():
        return
    r = pd.read_csv(p)
    order = ["ninguna", "mp3_128", "mp3_64", "noise_30", "noise_20", "lp_6k", "lp_4k",
             "speed-2", "speed+2", "speed-5", "speed+5", "pitch-1", "pitch+1", "pitch-2", "pitch+2",
             "speed+3", "pitch+0.5"]
    names = {"ninguna": "sin cambios", "mp3_128": "MP3 128 kbps", "mp3_64": "MP3 64 kbps", "noise_30": "ruido (SNR 30 dB)",
             "noise_20": "ruido (SNR 20 dB)", "lp_6k": "pasabajos 6 kHz", "lp_4k": "pasabajos 4 kHz",
             "speed-2": "velocidad −2%", "speed+2": "velocidad +2%", "speed-5": "velocidad −5%", "speed+5": "velocidad +5%",
             "pitch-1": "tono −1 semitono", "pitch+1": "tono +1 semitono", "pitch-2": "tono −2 semitonos",
             "pitch+2": "tono +2 semitonos", "speed+3": "velocidad +3% (*)", "pitch+0.5": "tono +½ semitono (*)"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 6), sharey=True)
    y = np.arange(len(order))
    for ax, col, title in [(axes[0], "real", "Reales bien clasificadas (%)"), (axes[1], "IA (total)", "IA detectada (%)")]:
        for model, color, off in [("original", "#2a78d6", -0.18), ("con augmentation", "#eb6834", 0.18)]:
            s = r[r["modelo"] == model].set_index("transformación").loc[order, col] * 100
            ax.barh(y + off, s.values, height=0.34, color=color, label=f"modelo {model}")
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [names[t] for t in order])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8.5)
    for ax in axes:
        ax.axhline(len(order) - 2.5, color=TEXT2, lw=0.6, ls=":")
    fig.suptitle("Robustez: qué pasa si se modifica el audio antes de analizarlo", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.01, "(*) magnitudes que el modelo con augmentation no vio al entrenar", fontsize=8, color=TEXT2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGS / "fig5_robustez.png")
    plt.close(fig)


def fig_pitch(band):
    """Por qué cambiar tono/velocidad rompe el detector: los picos de Suno se corren de lugar.

    Se usa el cambio de velocidad (+5%) porque es un resampleo puro; el pitch shift de librosa
    agrega además un ondulado propio (phase vocoder) que ensucia el gráfico.
    """
    m0, s0 = load_view("canon")
    m1, s1 = load_view("canon", "speed+5")
    ev = pd.read_csv(RES / "folds.csv").query("fold == 0")["uid"]
    f = FREQS[band_mask(band)] / 1000
    w = (f >= 2.9) & (f <= 3.3)
    fig, ax = plt.subplots(figsize=(9, 3))
    for mm, ss, col, lab in [(m0, s0, "#eb6834", "Suno v3.5 original"), (m1, s1, "#6250d6", "Suno v3.5 acelerado 5%")]:
        sel = ((mm["generator"] == "suno_v3.5") & mm["uid"].isin(ev)).values
        ax.plot(f[w], X_of(ss[sel], band)[:, w].mean(0), color=col, label=lab, lw=1.2)
    ax.axvline(3.0, color=TEXT2, lw=0.6, ls=":")
    ax.axvline(3.0 * 1.05, color=TEXT2, lw=0.6, ls=":")
    ax.set_xlabel("frecuencia (kHz)")
    ax.set_ylabel("altura del pico")
    ax.legend(loc="upper right")
    ax.set_title("Acelerar 5% mueve el pico de 3,00 a 3,15 kHz: el modelo lo busca en el lugar viejo")
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_corrimiento.png")
    plt.close(fig)


def fig_peaks_lasso():
    """¿El modelo usa los picos? Picos detectados vs bins elegidos por el Lasso."""
    p = RES / "peaks_final.npz"
    if not p.exists():
        return
    z = np.load(p)
    f, P, suno = z["f"] / 1000, z["P"], z["suno"]
    best = json.load(open(RES / "peaks_summary.json"))
    w_sparse = np.load(RES / "lasso_w_C0.01.npy")
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    d = z["mean_suno"] - z["mean_real"]
    axes[0].plot(f, d, color="#eb6834", lw=0.6)
    axes[0].plot(f[suno], d[suno], "o", ms=3, color=TEXT2, label=f"picos detectados ({suno.sum()})")
    axes[0].set_title("Huella promedio de Suno menos la de las reales, con los picos detectados", fontsize=10)
    axes[0].legend(loc="upper right")
    for ax, w, title in [
        (axes[1], w_sparse, "Lasso muy esparso (λ = 100): los pocos bins que usa para decir IA son picos"),
        (axes[2], z["w1"], f"Lasso elegido por CV (λ = {best['lasso_lambda']:.1f}): suma cientos de bins que no son picos"),
    ]:
        nz = w != 0
        col = np.where(P[nz], "#eb6834", np.where(w[nz] > 0, "#e8a37f", "#2a78d6"))
        ax.vlines(f[nz], 0, w[nz], colors=col, lw=0.8)
        ax.axhline(0, color=TEXT2, lw=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("coeficiente")
    from matplotlib.lines import Line2D
    axes[2].legend(handles=[Line2D([], [], color="#eb6834", lw=2, label="en un pico"),
                            Line2D([], [], color="#e8a37f", lw=2, label="indica IA, fuera de picos"),
                            Line2D([], [], color="#2a78d6", lw=2, label="indica real")],
                   loc="lower right", ncols=3, fontsize=8)
    axes[2].set_xlabel("frecuencia (kHz)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_lasso_picos.png")
    plt.close(fig)


def fig_peaks_retrain():
    """Reentrenar solo con picos / solo con bins al azar / sin picos, por clase."""
    p = RES / "peaks_retrain.csv"
    if not p.exists():
        return
    r = pd.read_csv(p).groupby("caso")[["real", "suno_v3.5", "udio_130", "n_bins"]].mean()
    cases = [("todos los bins", "todos los bins", "#b9b8b3"), ("solo picos", "solo los picos", "#eb6834"),
             ("solo bins al azar", "solo bins al azar (misma cantidad)", "#6250d6"), ("sin picos", "todo menos los picos", "#2a78d6")]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    h = 0.2
    for i, (key, lab, col) in enumerate(cases):
        yv = np.arange(3) + (i - 1.5) * h
        v = r.loc[key, GENS].values * 100
        ax.barh(yv, v, height=h - 0.03, color=col, label=f"{lab} ({int(round(r.loc[key, 'n_bins']))} bins)")
        for yy, vv in zip(yv, v):
            ax.text(vv + 0.5, yy, f"{vv:.0f}", va="center", fontsize=7.5, color=TEXT2)
    ax.set_yticks(np.arange(3), [LABELS[g] for g in GENS])
    ax.invert_yaxis()
    ax.set_xlim(40, 104)
    ax.set_xlabel("% bien clasificadas (validación cruzada, modelo reentrenado con esos bins)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncols=2, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_title("Los picos alcanzan para Suno, pero no sirven para Udio ni son necesarios")
    fig.tight_layout()
    fig.savefig(FIGS / "fig8_picos_reentrenar.png")
    plt.close(fig)


def fig_ablation_curve():
    """Ablación con umbral decreciente: picos pisados vs misma cantidad de bins al azar (modelo ya entrenado)."""
    p = RES / "peaks_ablation_curve.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    base = d[d["caso"] == "sin tocar"][["real", "suno_v3.5", "udio_130"]].mean() * 100
    g = d[d["caso"] != "sin tocar"].groupby(["caso", "K"])[["n_bins", "real", "suno_v3.5", "udio_130"]].mean().reset_index()
    n_band = len(FREQS[band_mask(choice()[0])])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, col, title in [(axes[0], "suno_v3.5", "Suno v3.5 detectado"), (axes[1], "udio_130", "Udio 130 detectado"),
                           (axes[2], "real", "Reales bien clasificadas")]:
        for caso, color, lab in [("picos", "#eb6834", "picos pisados"), ("azar", "#6250d6", "misma cantidad al azar")]:
            gg = g[g["caso"] == caso].sort_values("n_bins")
            x = np.r_[0, 100 * gg["n_bins"].values / n_band]
            ax.plot(x, np.r_[base[col], gg[col].values * 100], "-o", ms=4, lw=1.6, color=color, label=lab)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("% de la banda pisada")
        ax.set_ylim(50, 101)
    gp = g[g["caso"] == "picos"].sort_values("n_bins")
    for _, r in gp.iterrows():
        if r["K"] in (10, 5, 3, 2):
            axes[0].annotate(f"umbral {r['K']:g}", (100 * r["n_bins"] / n_band, r["suno_v3.5"] * 100),
                             textcoords="offset points", xytext=(4, -12), fontsize=7.5, color=TEXT2)
    axes[0].set_ylabel("%")
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Pisar picos (cada vez con umbral más bajo) vs pisar bins al azar, en el modelo ya entrenado",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "fig9_ablacion_curva.png")
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    band, C = choice()
    m, spec = load_view("canon")
    X = X_of(spec, band)
    if (RES / "cv_codec.csv").exists():
        fig_accuracy()
    fig_robustness()
    fig_peaks_lasso()
    fig_peaks_retrain()
    fig_ablation_curve()
    if (RES / "robustness.csv").exists():
        fig_pitch(band)
    fig_codec(m, spec)
    fig_fingerprints(m, X, band)
    fig_zoom(m, X, band)
    f, w = fig_weights(m, X, band, C)
    top = np.argsort(w)[::-1][:15]
    print("frecuencias con más peso hacia IA (Hz):", np.round(np.sort(f[top] * 1000), 1))


if __name__ == "__main__":
    main()
