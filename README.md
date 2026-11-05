# Detección de música generada por IA mediante análisis espectral

Trabajo práctico — Procesamiento de Señales / Series Temporales

## Idea

Los modelos generativos de música (Suno, Udio, MusicGen, etc.) sintetizan el audio con un decoder neuronal cuyas capas de *upsampling* (deconvolución con stride) introducen artefactos periódicos en el dominio de la frecuencia: picos espectrales pequeños pero sistemáticos en posiciones fijas, análogos a los artefactos *checkerboard* en imágenes [1]. La música grabada por humanos no presenta esta estructura. El proyecto consiste en construir un detector de música generada por IA basado exclusivamente en herramientas de procesamiento de señales (STFT, PSD, detección de picos), sin redes neuronales, y evaluar sus límites.

## Objetivos

1. **Caracterizar el artefacto.** Calcular la PSD promedio (Welch) y el espectro promediado en el tiempo (STFT) de música real y generada, restar el piso local y visualizar los picos periódicos y el corte de ancho de banda que distinguen a la música IA.
2. **Construir un detector interpretable.** A partir de la "huella" espectral de cada fragmento, extraer features (posición y altura de picos, roll-off, energía en bandas altas, planitud espectral) y entrenar un clasificador lineal como *baseline*, reproduciendo el criterio de [1].
3. **Evaluar robustez.** Medir la degradación del detector bajo transformaciones que alteran el espectro (remuestreo a 22.05 kHz, filtrado pasabajos, compresión MP3, pitch-shift), siguiendo la metodología de [2] y [3], y verificar si el detector aprende el artefacto o el pipeline de producción.
4. **Explorar mejoras.** Comparar modelos de ML (regresión logística, SVM, gradient boosting) y features derivadas invariantes a la posición de los picos (p. ej. autocorrelación o cepstrum de la huella) para mejorar la generalización a transformaciones y a generadores no vistos en entrenamiento.

## Datos

- Música generada: dataset **SONICS** [4] (Suno, Udio) y/o **FakeMusicCaps** [5].
- Música real: subconjunto de **FMA** (Free Music Archive).
- Todo el audio se homogeneiza a 44.1 kHz, mono y mismo formato para evitar que el clasificador aprenda diferencias de codec en lugar del artefacto.


## Explicaciones

### PSD (power spectral density)
PSD es la densidad espectral de potencia: una curva que te dice cuánta potencia tiene la señal en cada frecuencia. Es lo que calcula Welch: para una canción te devuelve un vector con un valor por frecuencia, graficado en dB.

“PSD promedio” se refiere a promediar esas curvas entre muchas canciones de la misma clase. Calculás la PSD de cada fragmento real, las promediás todas y obtenés una sola curva que representa “cómo es el espectro típico de la música real”. Hacés lo mismo con las canciones de IA, y superponés las dos.

El motivo de promediar es que la PSD de una sola canción depende mucho de esa canción (género, instrumentación, mezcla), y eso enmascara lo que te interesa. Al promediar sobre cientos de canciones, las diferencias individuales se cancelan y queda lo que es sistemático en cada clase: en la música IA, el corte abrupto de ancho de banda y los picos fijos del decoder, que en la real no están.


### STFT (Short-Time Fourier Transform)
Se parte la señal en ventanas cortas solapadas, se aplica una FFT a cada una y se obtiene una matriz tiempo-frecuencia (el espectrograma). Sirve para ver cómo cambia el contenido en frecuencia a lo largo de la canción, a diferencia de la PSD de Welch, que resume toda la señal en un único espectro.


## Referencias

[1] D. Afchar, G. Meseguer-Brocal, K. Akesbi, R. Hennequin. *A Fourier Explanation of AI-music Artifacts*. ISMIR 2025. https://arxiv.org/abs/2506.19108 — Código: https://github.com/deezer/ismir25-ai-music-detector

[2] L. Cros Vila, B. L. T. Sturm. *The AI Music Arms Race: On the Detection of AI-Generated Music*. Transactions of the ISMIR, 2025. https://transactions.ismir.net/articles/10.5334/tismir.254

[3] D. Afchar, G. Meseguer-Brocal, R. Hennequin. *Detecting music deepfakes is easy but actually hard*. 2024. https://arxiv.org/abs/2405.04181

[4] M. A. Rahman et al. *SONICS: Synthetic Or Not — Identifying Counterfeit Songs*. ICLR 2025. https://arxiv.org/abs/2408.14080

[5] L. Comanducci et al. *FakeMusicCaps: a Dataset for Detection and Attribution of Synthetic Music Generated via Text-to-Music Models*. 2024. https://arxiv.org/abs/2409.10684
