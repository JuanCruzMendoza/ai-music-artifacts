# Detección de música generada por IA mediante análisis espectral

Trabajo práctico — Procesamiento de Señales / Series Temporales

## Idea

Los generadores de música (Suno, Udio, etc.) producen el audio con un decoder neuronal. Sus capas de *upsampling* (deconvolución con stride) dejan picos espectrales chicos pero sistemáticos, siempre en las mismas frecuencias, análogos a los artefactos *checkerboard* en imágenes [1]. La música grabada por humanos no tiene esa estructura.

El TP replica la Tabla 2 de [1]: un clasificador lineal sobre esa "huella" espectral, entrenado con canciones de SONICS (IA) y FMA (reales). Además mide sus límites: ¿detecta el artefacto o el formato de los archivos? ¿Sobrevive a modificaciones del audio?

## Hipótesis

1. La huella espectral alcanza para separar Suno y Udio de música real con un clasificador lineal.
2. La huella se rompe con transformaciones que mueven las frecuencias (cambio de tono o de velocidad) y resiste las que no las mueven (MP3, pasabajos, ruido).

## Diseño experimental

Detalle completo en [docs/spec.md](docs/spec.md).

**Datos**

| Clase | Fuente | Generadores |
|---|---|---|
| Real | FMA-small (clips de 30 s) | — |
| IA | SONICS [4] | Suno v2, v3, v3.5 · Udio 32, 130 |

**Particiones.** Dev y heldout son disjuntas (sin artistas de FMA ni canciones de SONICS compartidas).
- **dev:** con dev se elige todo por validación cruzada de 5 folds. Contiene ~7.200 reales y ~1.800 canciones por generador visto (Suno v3.5 y Udio 130).
- **heldout:** se evalúa una sola vez, con el pipeline congelado. Tiene dos partes:
  - generadores vistos (Suno v3.5, Udio 130);
  - generadores no vistos (Suno v2, Suno v3, Udio 32).

**Preprocesamiento.** SONICS viene en un MP3 muy comprimido (16 kHz, mono, ~38 kbps). Para que el clasificador no aprenda el formato del archivo en lugar del artefacto, las reales se re-comprimen igual. Todo se recorta a la misma duración.

**Features.** La huella espectral del código de Deezer [1]:
1. Espectro promedio con FFT de 16.384 puntos (≈ 1 Hz por bin).
2. Se resta el piso local, para que solo queden los picos.
3. Se normaliza.

No se usan otras features.

**Modelo.** Regresión logística (como el paper), comparada con SVM lineal y SVM RBF. Métrica principal: *balanced accuracy*.

**Experimentos**

| | Qué se mide |
|---|---|
| E1 | Cómo es la huella de cada clase y qué frecuencias usa el clasificador |
| E2 | Acierto por clase (comparable con la Tabla 2 del paper) |
| Control de codec | E2 sin igualar el MP3, como el paper: ¿cuánto del acierto viene del formato? |
| E3 | Robustez: tono ±1/±2 semitonos, velocidad ±2/±5%, MP3 64/128 kbps, pasabajos 6/4 kHz, ruido SNR 30/20 dB |
| E4 | Si entrenar con audio modificado (augmentation) mejora la robustez |

## Documentación

- [docs/spec.md](docs/spec.md): diseño del experimento.
- [docs/log.md](docs/log.md): decisiones de implementación y cambios respecto del spec.
- [docs/resultados.md](docs/resultados.md): hallazgos (por ahora solo dev).

## Cómo correr

```bash
uv sync
uv run python scripts/01_manifest.py   # define dev / heldout
uv run python scripts/02_download.py   # baja solo dev (~11 GB)
uv run python scripts/03_spectra.py    # espectros
uv run python scripts/04_cv.py         # E2 + control de codec
```

Los pasos de robustez y figuras están en [docs/log.md](docs/log.md). El heldout (`scripts/07_heldout.py`) exige `--allow-heldout`.

## Referencias

[1] D. Afchar, G. Meseguer-Brocal, K. Akesbi, R. Hennequin. *A Fourier Explanation of AI-music Artifacts*. ISMIR 2025. https://arxiv.org/abs/2506.19108 — Código: https://github.com/deezer/ismir25-ai-music-detector

[2] L. Cros Vila, B. L. T. Sturm. *The AI Music Arms Race: On the Detection of AI-Generated Music*. Transactions of the ISMIR, 2025. https://transactions.ismir.net/articles/10.5334/tismir.254

[3] D. Afchar, G. Meseguer-Brocal, R. Hennequin. *Detecting music deepfakes is easy but actually hard*. 2024. https://arxiv.org/abs/2405.04181

[4] M. A. Rahman et al. *SONICS: Synthetic Or Not — Identifying Counterfeit Songs*. ICLR 2025. https://arxiv.org/abs/2408.14080
