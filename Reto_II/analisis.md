# Reto II -- Analisis y Reflexiones
**Series Temporales: Prediccion de Produccion Industrial**

**Autora:** Ana Lorena Jimenez Preciado
**Curso:** Machine Learning e Inteligencia Artificial -- Colegio de Matematicas Bourbaki

---

## 1. Resumen del Problema

El objetivo es predecir la produccion semanal agregada de dos grupos de activos
industriales (83 activos en total) a partir de mediciones diarias de sensores
(4 tipos de medida x 7 dias por semana). Los datos provienen del Data Challenge
de Kayrros/ENS (2020).

La dificultad radica en que:
- El target esta agregado por grupo, no por activo individual.
- Las mediciones son diarias pero el target es semanal.
- 29.5% de las mediciones son valores faltantes.
- Solo hay 104 semanas de datos, lo que limita modelos complejos.

---

## 2. Conceptos Clave

| Concepto | Definicion |
|---|---|
| **Serie temporal** | Secuencia de datos ordenados cronologicamente, donde el orden importa para el analisis. |
| **Formato tidy** | Estructura de datos donde cada fila es una observacion y cada columna una variable. |
| **Feature engineering** | Proceso de crear nuevas variables a partir de los datos crudos para mejorar el desempeno del modelo. |
| **Split temporal** | Division de datos donde el set de validacion contiene las semanas mas recientes, respetando la causalidad. |
| **MSE** | Error cuadratico medio: metrica que penaliza errores grandes cuadraticamente. |
| **GradientBoosting** | Ensemble de arboles de decision entrenados secuencialmente, cada uno corrigiendo errores del anterior. |
| **LSTM** | Long Short-Term Memory: red recurrente con compuertas que permite capturar dependencias temporales largas. |
| **Deteccion de anomalias** | Identificacion de patrones que se desvian significativamente del comportamiento normal. |

---

## 3. Metodologia

### 3.1 Pipeline

```
CSVs (tidy) --> Pivot (activo-semana) --> Features --> Modelo --> Prediccion --> Agregacion por grupo
```

### 3.2 Feature Engineering

Transformacion del formato tidy (241K filas) a matriz plana:

| Tipo de Feature | Cantidad | Descripcion |
|---|---|---|
| Mediciones brutas | 28 | mt{j}_wd{l}: tipo j, dia l |
| Estadisticas por tipo | 16 | media, std, max, min por tipo |
| Capacidad nominal | 1 | Capacidad del activo |
| Agregados por grupo | 25 | Estadisticas globales por semana |

### 3.3 Modelos Implementados

#### Baseline: Deteccion de Anomalias (On/Off)

Sugerido por el challenge:
- Calcula el centroide de mediciones por activo (comportamiento "normal").
- Si la distancia Z-score al centroide supera un umbral, el activo se clasifica "off" (produccion = 0).
- Si esta cerca del centroide, se clasifica "on" (produccion = capacidad nominal).
- La produccion del grupo = suma de capacidades de activos "on".

#### GradientBoosting Regressor (GBR)

- Un modelo por grupo que predice produccion semanal directamente.
- Features: estadisticas agregadas de mediciones de todos los activos del grupo.
- Hiperparametros: 300 arboles, max_depth=5, lr=0.05.

#### LSTM (Comparacion)

- Red recurrente con ventana deslizante de 4 semanas.
- 2 capas LSTM (64 unidades) + capa densa.
- 100 epocas con Adam (lr=1e-3).
- **Limitacion:** Con solo ~80 semanas de entrenamiento, este modelo sufre de
  sobreajuste severo. Se incluye como comparacion pedagogica para demostrar
  que modelos mas complejos no necesariamente superan a los simples cuando
  los datos son escasos.

---

## 4. Resultados

### 4.1 Comparacion de Modelos

| Modelo | Grupo | MSE | RMSE |
|---|---|---|---|
| **BASELINE** | **2** | **3.42e+10** | **184,936** |
| BASELINE | 3 | 5.59e+11 | 747,381 |
| GBR | 2 | 6.54e+10 | 255,801 |
| **GBR** | **3** | **2.36e+11** | **485,510** |
| LSTM | 2 | 1.51e+11 | 389,029 |
| LSTM | 3 | 2.74e+11 | 523,797 |

### 4.2 Observaciones

1. **El baseline gana en el Grupo 2:** Un modelo simple de "on/off" captura
   bien el comportamiento del Grupo 2, donde los activos tienen patrones
   mas binarios. Esto confirma la intuicion del challenge: para activos
   relativamente estables, detectar cuando estan "apagados" es suficiente.

2. **GBR gana en el Grupo 3:** El Grupo 3, con mas activos y mayor variabilidad,
   se beneficia de un modelo que puede capturar factores de capacidad continuos
   en lugar de binarios.

3. **LSTM es el peor modelo en ambos grupos:** Con solo 80 semanas de
   entrenamiento y secuencias de 4 timesteps, el LSTM no tiene suficientes
   datos para aprender patrones temporales significativos. Las curvas de
   aprendizaje muestran sobreajuste: el loss de entrenamiento baja pero
   el de validacion se estanca o sube.

4. **El problema es inherentemente dificil:** Incluso el mejor modelo tiene
   un RMSE de ~185K para el Grupo 2, lo cual representa ~5% de la produccion
   media (3.79M). Para el Grupo 3, el RMSE del GBR es ~486K, ~5.3% de
   la media (9.14M).

### Graficas Generadas

Todas las graficas se encuentran en la carpeta `resultados/`:

| Archivo | Contenido |
|---|---|
| `01_produccion_semanal.png` | Series de produccion por grupo |
| `02_capacidades_nominales.png` | Distribucion de capacidades |
| `03_nan_heatmap.png` | Valores faltantes por activo/medida |
| `04_distribucion_medidas.png` | Distribucion de cada tipo de medida |
| `05_predicciones.png` | Real vs estimado (3 modelos x 2 grupos) |
| `06_comparacion_mse.png` | Barplot comparativo de MSE |
| `07_feature_importance.png` | Features mas importantes (GBR) |
| `08_lstm_learning.png` | Curvas de aprendizaje LSTM |
| `09_resumen_resultados.csv` | Tabla resumen en CSV |

---

## 5. Preguntas de Reflexion

### Por que el baseline simple funciona bien para ciertos grupos?

Porque algunos activos industriales operan de forma esencialmente binaria:
estan produciendo a capacidad nominal o estan apagados por mantenimiento/incidente.
Cuando el comportamiento real se aproxima a este patron, la deteccion de anomalias
es un enfoque natural y robusto.

### Cuando conviene un modelo mas complejo (GBR, LSTM)?

Cuando los activos no son simplemente on/off sino que operan a niveles parciales
de capacidad (e.g., 40%, 70%, 100%). El GBR puede aprender estos factores de
capacidad continuos a partir de las mediciones. El LSTM seria apropiado si
hubiera suficientes datos (~500+ semanas) para capturar dependencias temporales.

### Por que el LSTM no funciona con pocos datos?

1. **Parametros vs muestras:** Con ~80 secuencias de entrenamiento y miles de
   parametros, la relacion datos/parametros es muy desfavorable.
2. **Secuencias cortas:** Con solo 4 timesteps por secuencia, el LSTM no puede
   capturar patrones estacionales largos.
3. **Normalizacion inestable:** Con tan pocos datos, las estadisticas de
   normalizacion (media, std) del train pueden no representar bien el val.

Este es un caso pedagogico importante: **un modelo mas complejo no es
necesariamente mejor.** La eleccion del modelo debe considerar la cantidad
de datos disponible.

### Como mejorar los resultados?

- **Incorporar variables exogenas:** precios de energia, temperatura, demanda
  del mercado, dias festivos.
- **Modelo hibrido:** Usar el baseline para detectar off/on y luego un regresor
  para estimar el factor de capacidad parcial.
- **Cross-validation temporal:** Usar rolling window en lugar de un solo split.
- **Tuning de hiperparametros:** Grid search sobre threshold del baseline y
  parametros del GBR.

---

## 6. Estructura del Proyecto

```
Reto_II/
|-- main.py                 # Punto de entrada
|-- analisis.md             # Este archivo
|-- flashcard.md             # Data Card
|-- Reto_II_ML_AI.pdf       # Enunciado del reto
|-- data/
|   |-- X_train.csv         # Mediciones diarias
|   |-- y_train.csv         # Produccion semanal
|   +-- assets.csv          # Capacidades nominales
|-- src/
|   |-- __init__.py
|   |-- config.py           # Configuracion global
|   |-- data_loader.py      # Carga y transformacion
|   |-- features.py         # Feature engineering
|   |-- model.py            # Tres modelos (baseline, GBR, LSTM)
|   |-- training.py         # Entrenamiento y evaluacion
|   +-- visualization.py    # Graficas
+-- resultados/             # Graficas y CSVs generados
```

---

*Elaborado por Ana Lorena Jimenez Preciado -- Curso ML e AI, Colegio de Matematicas Bourbaki*
