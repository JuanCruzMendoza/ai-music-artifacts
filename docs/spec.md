# Spec del experimento

Réplica de la Tabla 2 de Afchar et al. (ISMIR 2025): SONICS vs FMA, más una evaluación de robustez.

## Hipótesis

1. La huella espectral (los picos del decoder) alcanza para separar la música de Suno y Udio de la música real usando un clasificador lineal.
2. Esa huella se rompe con las transformaciones que mueven las frecuencias (pitch shift, cambio de velocidad), y es más estable ante las que no las mueven (MP3, pasabajos, ruido).

## Datos

| Fuente | Clase | Detalle |
|---|---|---|
| FMA-small | real | 8.000 clips de 30 s, 8 géneros balanceados. MP3 de 128 a 320 kbps (la mayoría a 320), mayormente a 44,1 kHz estéreo |
| SONICS (`fake_songs.csv`) | fake | 49.074 canciones. **MP3 a 16 kHz, mono, LAME 3.100 `-V 4` (~38 kbps VBR)**. Generadores: Suno v2, v3, v3.5 (`chirp-*`) y Udio 32, 130 (`udio-30s`, `udio-120s`) |

SONICS no distribuye las canciones reales, solo IDs de YouTube. Por eso las reales salen de FMA, igual que en el paper.

El formato de SONICS se verificó en 2 archivos; la columna `bit_rate` (entre 22 y 57k, mediana ~37k) es consistente con eso en todo el dataset.

## Splits

Hay dos particiones disjuntas:

- **dev:** se usa para entrenar y elegir todo (banda, modelo, hiperparámetros) mediante 5-fold CV.
- **heldout:** se evalúa **una sola vez**, con el pipeline congelado.

| Partición | Real (FMA) | Fake (SONICS) |
|---|---|---|
| dev | split oficial `training` + `validation` | `split == train`, solo Suno v3.5 y Udio 130 |
| heldout-seen | split oficial `test` | `split == test`, Suno v3.5 y Udio 130 |
| heldout-unseen | (mismas reales que heldout-seen) | `split == test`, Suno v2, Suno v3 y Udio 32 |

- **FMA:** se usa la columna `('set', 'split')` de `tracks.csv`. En small hay 6.400 / 800 / 800 clips (training / validation / test), sin artistas compartidos entre splits (verificado).
- **SONICS:** se respeta la columna `split`. La partición `valid` no se usa. Las dos versiones de una misma canción (`_0` y `_1`) quedan juntas y ningún par (letra, estilo) cruza particiones.

Fakes disponibles por generador:

| `algorithm` | train | test | Duración (mediana) |
|---|---|---|---|
| `chirp-v3.5` (Suno v3.5) | 17.195 | 1.590 | 203 s |
| `udio-120s` (Udio 130) | 16.828 | 1.619 | 131 s |
| `chirp-v2-xxl-alpha` (Suno v2) | 0 | 1.759 | 80 s |
| `chirp-v3` (Suno v3) | 0 | 3.637 | 120 s |
| `udio-30s` (Udio 32) | 0 | 4.173 | 33 s |

- **Tamaños:** dev con 7.200 reales y 1.800 fakes por generador visto (muestreo aleatorio con semilla fija); heldout con 800 reales y 400 fakes por generador.
- **Descarga:** SONICS son 10 zips de unos 3,3 GB cada uno, y cada zip es un archivo independiente. Se bajan solo los archivos elegidos, con requests HTTP por rango sobre el zip remoto (verificado). Son unos 5.600 archivos.
- **Clases desbalanceadas:** se usa `class_weight="balanced"`.

## Preprocesamiento

Las dos clases tienen que terminar en el mismo formato:

1. **Igualar el codec de FMA al de SONICS:** decodificar el MP3, pasar a mono, resamplear a 16 kHz y re-encodear con LAME 3.100 `-V 4`. Sin este paso, el clasificador puede aprender "MP3 de bajo bitrate", y el paper no lo hace.
2. Para las dos clases: decodificar el MP3 a 16 kHz mono.
3. Recortar a los **primeros 30 s** y descartar los clips más cortos. FMA ya mide 30 s, mientras que SONICS dura entre 32 y 240 s. Si no se recorta, las fakes promedian más tiempo y su huella sale menos ruidosa, lo que sería una confusión.

## Features: la huella espectral

Siguen `compute_fakeprints.py` de Deezer:

1. STFT con `n_fft = 16384` (≈ 1 Hz por bin), ventana Hann y hop de 8.192.
2. Pasar a dB y promediar en el tiempo.
3. Restar la envolvente inferior (mínimo local en ventanas de 10 bins, interpolada) y recortar a 0.
4. Recortar a [0, 5] dB y normalizar por el máximo.
5. Banda de **1 a 8 kHz**, unos 7.000 bins.

Solo se usan las features de la huella. No se usan features de ancho de banda ni de roll-off.

## Modelos

- **Baseline:** `LogisticRegression(class_weight="balanced")`, eligiendo `C` por CV.
- **Comparación:** SVM lineal y SVM RBF. Gradient boosting es opcional.

## Experimentos

| ID | Qué | Dónde |
|---|---|---|
| E1 | Huella promedio por clase y por generador, y pesos de la LogReg | dev |
| E2 | Accuracy por clase (como la Tabla 2), balanced accuracy y ROC-AUC | CV en dev, luego heldout-seen y heldout-unseen |
| E3 | Robustez: el modelo final evaluado sobre heldout transformado | heldout |
| E4 *(opcional)* | Reentrenar con dev aumentado (las transformaciones de E3) y repetir E3 | dev, luego heldout |

Transformaciones de E3, aplicadas al audio **antes** del preprocesamiento:

- Pitch shift de ±1 y ±2 semitonos.
- Cambio de velocidad por resampleo de ±2% y ±5%, que escala las frecuencias.
- Re-encodeo a MP3 de 64 y 128 kbps.
- Pasabajos a 6 y 4 kHz.
- Ruido blanco con SNR de 30 y 20 dB.

## Controles

- **Control de codec:** repetir E2 sin igualar el codec (FMA solo resampleado, como en el paper) y comparar. Si el accuracy cambia, el codec estaba influyendo.
- Verificar que los pesos de la LogReg se concentren en los picos y no en otra estructura.
- Reportar la CV en dev con su media ± desvío.

## Puntos abiertos

- **Banda:** el paper dice 1–8 kHz para SONICS, pero el código usa 5–16 kHz, que a 16 kHz queda en 5–8 kHz. El pasabajos de LAME en SONICS está en 8 kHz. Decidir por CV en dev.
- **Inicio del recorte:** ¿saltear la intro de las canciones?
