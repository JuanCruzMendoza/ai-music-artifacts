# Log de implementación

Decisiones y cambios respecto de [spec.md](spec.md), en orden. Todo se corrió **solo con dev**: el heldout no se bajó ni se procesó.

## Datos y descarga
- **Descarga selectiva.** Se bajan solo los archivos de dev, leyendo cada uno del zip remoto con requests HTTP por rango (`src/remotezip.py`). El zip de FMA comprime con bzip2. Todo dev pesa ~11 GB.
- **Barrera para el heldout.** El manifiesto (`data/meta/manifest.csv`) ya define qué archivos van al heldout, pero los scripts filtran `partition == dev` salvo que se pase `--allow-heldout`.
- **Suno v3.5 en dev:** quedaron 1.801 archivos, no 1.800, porque se muestrean canciones completas (las versiones `_0` y `_1` juntas).

## Preprocesamiento
- **Recorte a 29 s, no 30.** Al decodificarlos, el 57% de los clips de FMA dura 29,98 s (medido en 300 clips), así que con 30 s exactos se descartaba más de la mitad.
- **SONICS se toma desde el segundo 0,** sin saltear la intro. Es lo más simple, y la huella no depende del contenido musical.
- **Igualación de codec.** FMA pasa a mono, se resamplea a 16 kHz con `soxr` y se re-encodea con ffmpeg/libmp3lame `-q:a 4`, que equivale a LAME `-V 4`. El resultado es ~36 kbps VBR a 16 kHz, igual que SONICS.
  - El header del archivo dice "Lavc" en vez de "LAME3.100", pero el encoder subyacente es el mismo.
  - Bug corregido: si ffmpeg escribe a un pipe, el header VBR queda mal y el decoder cortaba el audio a ~17 s. Ahora escribe a un archivo temporal.
- **STFT reimplementada en numpy** (`src/fingerprint.py`) para no instalar torch. Usa los mismos parámetros que `torchaudio.Spectrogram` del código de Deezer: `n_fft` 16384, hop 8192, ventana Hann, centrada, padding reflect. Las funciones `lower_hull`, `curve_profile` y `max_normalise` también replican las de Deezer.
- **Se guarda el espectro promedio en dB (0–8 kHz) de cada archivo,** no la huella. La huella se calcula después para cualquier banda, lo que permite elegir la banda sin recalcular el audio.

- **Archivos descartados:** 6 MP3 de FMA no se pueden decodificar (098565, 098567, 098569, 099134, 108925, 133297; son problemas conocidos de FMA). Dev queda con 7.194 reales, 1.801 de Suno v3.5 y 1.800 de Udio 130. Otros archivos de FMA emiten avisos de `mpg123` ("dequantization failed"), pero decodifican bien.
- **Rate limit de Hugging Face:** con 24 conexiones, 192 archivos de SONICS fallaron con error 429. Se reintentaron con 4 conexiones y bajaron todos.

## Validación cruzada
- **5 folds agrupados** (`StratifiedGroupKFold`): un mismo artista de FMA, o una misma canción de SONICS, nunca queda en dos folds distintos.
- **Banda y C se eligen con la misma CV en que se reportan los resultados,** así que los números de dev son algo optimistas. La medida limpia es el heldout.
- **Banda:** se agregaron bandas que terminan en 7,2 kHz, porque el MP3 `-V 4` a 16 kHz corta en ~7,3 kHz y deja un pico de borde en las tres clases.
  - Primera corrida: una regla de "empate → banda del paper" eligió 1–8 kHz, y los pesos más grandes del modelo caían en el borde del MP3 (7,25 kHz). Se cambió la regla a "mejor balanced accuracy", que eligió **0,5–7,2 kHz**.
  - La primera corrida quedó guardada en `data/results/cv_run1.log`.
- **C:** la grilla se amplió a 0,001–10 porque el mejor valor caía en el borde. Se eligió **C = 0,03**.
- **Control de codec:** se corrió en 4 combinaciones (igualado o no × banda nuestra o del paper).
  - El dato "FMA original vs FMA re-encodeado" con la banda 1–8 kHz (98,7%) viene de la primera corrida, con C = 0,01.
  - Con la banda 0,5–7,2 kHz da 53%; esa cifra viene de la corrida final.
- **Métrica principal: *balanced accuracy*** = (acierto en reales + acierto en IA) / 2, con Suno y Udio juntos como "IA". Se usa para elegir la banda y C, y para reportar. En robustez (E3/E4) se agregó como columna de `robustness.csv`.
- **Modelos extra:** SVM lineal (C = 0,1, sin ajustar) y SVM RBF (con estandarización y parámetros por defecto). No se ajustaron más para mantener el TP simple.

## Robustez (E3/E4)
- **El spec decía evaluar en el heldout; se hace dentro de dev.** El fold 0 (~20% de dev) se usa como evaluación y los folds 1–4 como entrenamiento. El script queda listo para correr sobre el heldout cuando se autorice.
- **Las transformaciones se aplican al audio ya recortado e igualado** (16 kHz mono, LAME `-V 4`), igual para reales y fakes. Así las dos clases pasan exactamente por lo mismo.
- **El cambio de velocidad acorta o alarga el audio** (±5% → ~27,6 a 30,5 s). La huella se promedia sobre lo que haya.
- **Augmentation (E4):** cada archivo de entrenamiento suma una copia con una transformación elegida al azar entre las 14 de E3. Por eso el modelo ve en el entrenamiento los mismos tipos de transformación que después se evalúan.
- **Control agregado: magnitudes no vistas.** Se evalúan además +0,5 semitonos y +3% de velocidad, que no están en el augmentation, para ver si la mejora de E4 generaliza. No generaliza.
- **El pitch shift de librosa** (phase vocoder) agrega un ondulado periódico propio a la huella. Por eso el gráfico del corrimiento de picos usa el cambio de velocidad, que es un resampleo puro.
- **E3/E4 usan solo la regresión logística.**

## ¿El modelo usa los picos? (`scripts/08_peaks.py`, agregado después del spec)
- **Definición de pico:** bins donde (huella media del generador − huella media de las reales) supera en más de 5 desvíos robustos (MAD × 1,4826) a su mediana local (ventana de 51 bins ≈ 50 Hz). Se usa la diferencia con las reales para no marcar lo que comparten todas las clases, como el borde de la banda o notas musicales comunes. Unión de Suno y Udio, ensanchada ±1 bin.
- **Sin fuga de datos:** dentro de la CV, los picos se recalculan en cada fold solo con los datos de entrenamiento. Las cifras de concentración y la Figura 7 usan los picos y modelos de todo dev.
- **Control al azar:** la misma cantidad exacta de bins que los picos, en bloques de 3 contiguos y fuera de los picos. Ablación: 20 repeticiones por fold. Reentrenar: 5 repeticiones por fold.
  - Bug corregido: la primera versión elegía la misma cantidad de *centros* y quedaba con más bins (377 contra 278).
- **Ablación:** los bins se pisan con los de una canción real de entrenamiento elegida al azar.
  - La primera versión usaba el promedio de las reales. Como los valores son asimétricos, el promedio queda por encima del valor típico y empujaba todo hacia "IA" (subía Udio y los falsos positivos).
- **Lasso:** `LogisticRegression(l1_ratio=1, solver="liblinear", class_weight="balanced")`. En sklearn 1.8 `penalty="l1"` está deprecado, por eso se usa `l1_ratio=1`. λ = 1/C, con C ∈ {0,003 … 10}, elegido por balanced accuracy en la misma CV de 5 folds. Resultado: **λ = 3,3 (C = 0,3)**, con balanced accuracy de 95,2%, algo menos que la regresión logística L2 (96,7%).
- **Hallazgo lateral:** algunos picos en 554 / 622 / 659 Hz aparecen en Suno y en Udio. Coinciden con notas musicales (Do#5, Re#5, Mi5): probablemente sean afinación perfecta de la IA, no artefactos del decoder.

## Heldout (preparado, no corrido)
- `scripts/07_heldout.py` exige `--allow-heldout`. Con ese flag, 02 y 03 procesan **solo** el heldout y guardan los espectros en archivos aparte (`*__heldout.npz`), así que dev y heldout no se mezclan.
- **Plan fijado de antemano:**
  - Banda y C de `choice.json`.
  - LogReg (principal) y SVM RBF entrenados en todo dev.
  - Control de codec con la banda del paper.
  - E3/E4 con los mismos modelos de `06_robustness.py`.

## Cómo reproducir (dev)
```bash
uv sync
uv run python scripts/01_manifest.py          # selección dev/heldout
uv run python scripts/02_download.py          # ~11 GB, solo dev
uv run python scripts/03_spectra.py           # codec igualado
uv run python scripts/03_spectra.py --base orig --source fma   # control de codec
uv run python scripts/04_cv.py                # E2 + control de codec -> data/results/choice.json
uv run python scripts/06_robustness.py prepare
for t in speed-5 speed-2 speed+2 speed+5 speed+3 mp3_128 mp3_64 lp_6k lp_4k noise_30 noise_20 pitch-2 pitch-1 pitch+1 pitch+2 pitch+0.5; do
  uv run python scripts/03_spectra.py --transform $t --uids data/results/robust_uids_$t.csv; done
uv run python scripts/06_robustness.py eval   # E3/E4
uv run python scripts/08_peaks.py             # ¿usa los picos? + Lasso
uv run python scripts/05_figures.py           # docs/figs/
```
