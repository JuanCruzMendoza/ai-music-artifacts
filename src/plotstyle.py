"""Estilo común de los gráficos (paleta validada: azul / naranja / aqua)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"real": "#2a78d6", "suno_v3.5": "#eb6834", "udio_130": "#1baf7a"}
LABELS = {"real": "Real (FMA)", "suno_v3.5": "Suno v3.5", "udio_130": "Udio 130"}
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
    "lines.linewidth": 1.2, "figure.dpi": 130,
})
