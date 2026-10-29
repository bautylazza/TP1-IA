# TP1 - Inteligencia Artificial: Redes Neuronales para Mantenimiento Predictivo

Trabajo Práctico N°1 de **Inteligencia Artificial** — Universidad Tecnológica Nacional, Facultad Regional Santa Fe (2026).

El proyecto diseña, entrena y evalúa una **red neuronal multicapa (MLP)** con TensorFlow/Keras para predecir la necesidad de mantenimiento de máquinas industriales. Continúa el trabajo del TP1 de Ciencia de Datos ([TP1-CDD](https://github.com/bautylazza/TP1-CDD)), reutilizando su dataset ya preprocesado y la misma división entrenamiento/validación.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bautylazza/TP1-IA/blob/main/TP1_IA.ipynb)

## Integrantes - Grupo 02

- Bautista Lazzarini
- Franco Mazzuchini
- Tomás Alzugaray
- Santiago Nicle

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `TP1_IA.ipynb` | Notebook con el diseño, los experimentos de hiperparámetros y la evaluación final. |
| `README.md` | Este documento. |

> El notebook lee `mantenimiento_procesado.csv`, que se genera en [TP1-CDD](https://github.com/bautylazza/TP1-CDD) y no está incluido en este repositorio. En Colab se carga desde Google Drive.

## Datos

Se usa el dataset de mantenimiento predictivo procesado en Ciencia de Datos: sin identificadores, con valores faltantes y erróneos tratados, `product_type` codificado con One-Hot, variables numéricas escaladas con `RobustScaler` y `target` binario (0 = normal, 1 = falla).

- División 75 % / 25 % (`random_state=42`, estratificada): **10.890 muestras de entrenamiento** y **3.631 de validación**.
- 7 variables de entrada.

## 1. Diseño del modelo

Arquitectura MLP **7 → 16 → 8 → 1**, con 273 parámetros entrenables:

| Capa | Neuronas | Activación | Justificación |
|---|---|---|---|
| Oculta 1 | 16 | ReLU | Captura interacciones entre las 7 variables de entrada. |
| Oculta 2 | 8 | ReLU | Estructura de "embudo" que consolida patrones y limita el sobreajuste. |
| Salida | 1 | Sigmoide | Clasificación binaria (probabilidad de falla). |

Compilado inicialmente con Adam y *binary crossentropy*, 30 épocas y batch size de 32.

## 2. Ajuste de hiperparámetros

Se parametrizó una función de entrenamiento (`ejecutar_experimento`) y se exploraron cuatro ejes, comparando las curvas de aprendizaje de cada combinación:

| Eje | Qué se probó | Mejor resultado (accuracy de validación) |
|---|---|---|
| **A - Optimizador y tasa de aprendizaje** | Adam, SGD y RMSprop × lr 0.01, 0.001, 0.0001 | Adam con lr = 0.01: 0.9405 |
| **B - Regularización** | Dropout (0, 0.2, 0.5) × L2 (0, 0.01, 0.001) | Sin regularización: 0.9375 |
| **C - Batch size y épocas** | Batch 16, 64, 256 × 20, 30, 60 épocas | Batch 64 con 60 épocas: 0.9380 |
| **D - Función de pérdida** | Binary crossentropy vs. error cuadrático medio | MSE: 0.9361 |

**Configuración elegida:** Adam (lr = 0.01), sin dropout ni L2, batch size 16, 30 épocas y error cuadrático medio como función de pérdida.

## 3. Evaluación del modelo final

Métricas sobre el conjunto de validación (3.631 muestras):

| Métrica | Valor |
|---|---|
| Accuracy | 0.9427 |
| Precision | 0.9293 |
| Recall | 0.9621 |
| F1-score | 0.9454 |
| AUC-ROC | 0.9824 |

El notebook incluye además las curvas de accuracy y pérdida por época, la curva ROC y la matriz de confusión. El rendimiento es comparable al de k-NN en el TP1 de Ciencia de Datos (accuracy 0.95) y superior al de la regresión logística (0.84), aunque inferior al de Gradient Boosting (0.98).

## Tecnologías

Python 3, Google Colab, TensorFlow / Keras, scikit-learn, pandas y matplotlib.

## Cómo ejecutarlo

1. Abrir el notebook en Google Colab con el botón de arriba.
2. Generar `mantenimiento_procesado.csv` ejecutando el notebook de [TP1-CDD](https://github.com/bautylazza/TP1-CDD) y subirlo a Google Drive.
3. Ajustar la ruta de lectura del CSV en el notebook y ejecutarlo de principio a fin.

Para correrlo en local, reemplazar el montaje de Drive por la lectura directa del archivo e instalar las dependencias:

```bash
pip install tensorflow scikit-learn pandas matplotlib seaborn jupyter
```
