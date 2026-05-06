# Reto III -- Analisis y Reflexiones
**Graph Neural Networks: Prediccion de Precios de Airbnb en Santorini**

**Autora:** Ana Lorena Jimenez Preciado
**Curso:** Machine Learning e Inteligencia Artificial -- Colegio de Matematicas Bourbaki

---

## 1. Resumen del Problema

El objetivo es predecir el precio por noche de listings de Airbnb en Santorini,
Grecia, utilizando Graph Neural Networks (GNN). Se construyen dos tipos de grafos
a partir de los datos: uno basado en proximidad geografica (KNN sobre coordenadas)
y otro basado en similitud textual (embeddings de sentence-transformers). La hipotesis
es que la informacion de vecinos en el grafo puede enriquecer las predicciones
respecto a un modelo tabular que solo ve cada listing de forma independiente.

La dificultad radica en que:
- Los precios tienen una distribucion muy asimetrica (media $358, mediana $180).
- Santorini tiene zonas de precios muy distintos (caldera vs interior de la isla).
- ~26% de listings no tienen review scores, reduciendo las features disponibles.

---

## 2. Conceptos Clave

| Concepto | Definicion |
|---|---|
| **Grafo** | Estructura matematica G=(V,E) compuesta por nodos V y aristas E que representan relaciones entre entidades. |
| **GNN (Graph Neural Network)** | Red neuronal que opera sobre grafos, propagando informacion entre nodos conectados mediante message passing. |
| **GCN (Graph Convolutional Network)** | Variante de GNN que aplica convoluciones espectrales sobre el grafo, promediando features de vecinos con normalizacion por grado. |
| **Message Passing** | Paradigma donde cada nodo actualiza su representacion agregando mensajes (features) de sus vecinos. |
| **Grafo KNN** | Grafo construido conectando cada nodo con sus K vecinos mas cercanos segun una metrica de distancia. |
| **Distancia Haversine** | Formula para calcular la distancia entre dos puntos sobre una esfera usando latitud y longitud. |
| **Sentence Embeddings** | Representaciones vectoriales densas de texto generadas por modelos como sentence-transformers. |
| **Similitud Coseno** | Metrica que mide el angulo entre dos vectores; valores cercanos a 1 indican alta similitud. |
| **Oversmoothing** | Problema de GNN donde demasiadas capas o grafos muy densos hacen que todas las representaciones converjan al mismo vector. |

---

## 3. Metodologia

### 3.1 Pipeline

```
Listings CSV --> Limpieza --> Features (23d) --> Grafo --> GCN --> Prediccion --> exp(pred)-1
                                             --> Embeddings texto
```

### 3.2 Preprocesamiento

| Etapa | Detalle |
|---|---|
| Filtrado | Eliminar precio <= 0 y outliers (percentil 99) |
| Features numericas | 16 variables: accommodates, beds, reviews, ubicacion, etc. |
| Features categoricas | One-hot de room_type (4 tipos) |
| Features binarias | is_superhost, instant_bookable |
| Imputacion | Mediana por columna para NaN |
| Estandarizacion | StandardScaler (media=0, std=1) |
| Target | log(precio + 1) para reducir asimetria |

### 3.3 Construccion de Grafos

#### Grafo Geografico (KNN, K=10)

- Se calcula la distancia haversine entre todos los pares de listings.
- Cada listing se conecta con sus 10 vecinos mas cercanos.
- Grafo resultante: 4,494 nodos, ~55,000 aristas.
- Intuicion: listings cercanos geograficamente tienden a tener precios similares.

#### Grafo Textual (Similitud Coseno, umbral=0.80)

- Se concatenan `name` y `description` de cada listing.
- Se generan embeddings con `all-MiniLM-L6-v2` (sentence-transformers, 384 dim).
- Se conectan pares de listings con similitud coseno > 0.80.
- Grafo resultante: 4,494 nodos, ~18,700 aristas.
- Intuicion: listings con descripciones similares (ej. "villa con vista a la caldera")
  compiten en el mismo segmento de mercado y tienen precios comparables.

### 3.4 Modelos

#### Baseline: Regresion Lineal

- Modelo tabular simple (sklearn LinearRegression).
- Cada listing se predice de forma independiente.
- No usa informacion del grafo.

#### GCN (Graph Convolutional Network)

Arquitectura de 2 capas:

```
GCNConv(23 -> 64) -> BatchNorm -> ReLU -> Dropout(0.3)
GCNConv(64 -> 64) -> BatchNorm -> ReLU -> Dropout(0.3)
Linear(64 -> 1)
```

- **Loss:** MSELoss (en espacio log-precio)
- **Optimizer:** Adam (lr=0.01, weight_decay=5e-4)
- **Early stopping:** paciencia de 30 epocas
- **Epocas maximas:** 300

---

## 4. Resultados

### 4.1 Comparacion de Modelos

| Modelo | MSE | RMSE | MAE | R2 |
|---|---|---|---|---|
| **LINEAR** | **56,806** | **238** | **127** | **0.5553** |
| GCN_GEO | 97,389 | 312 | 160 | 0.2376 |
| GCN_TXT | 76,152 | 276 | 137 | 0.4038 |

### 4.2 Observaciones

1. **La regresion lineal gana en todas las metricas.** Esto no es inusual:
   para datos tabulares con features bien disenadas, los modelos lineales son
   baselines muy competitivos. La GCN agrega complejidad (message passing) pero
   no necesariamente informacion nueva que las features ya no capturen.

2. **El grafo textual supera al geografico.** La similitud en descripciones
   (GCN_TXT, R2=0.40) captura mejor el segmento de mercado que la proximidad
   fisica (GCN_GEO, R2=0.24). Dos listings cercanos pueden tener precios muy
   distintos (hotel de lujo vs hostal), pero listings con descripciones similares
   ("stunning caldera view, private pool") probablemente estan en el mismo rango.

3. **Oversmoothing en GCN_GEO.** El grafo geografico con K=10 vecinos genera
   un grafo relativamente denso donde la GCN tiende a promediar precios de vecinos
   muy heterogeneos, perdiendo discriminacion.

4. **La transformacion log-precio mejora significativamente las predicciones.** Sin
   ella, el MSE estaba dominado por los outliers de alta gama. Con log-precio,
   el modelo se enfoca en las proporciones relativas.

### Graficas Generadas

Todas las graficas se encuentran en la carpeta `resultados/`:

| Archivo | Contenido |
|---|---|
| `01_distribucion_precios.png` | Histograma de precios (lineal y log) |
| `02_mapa_propiedades.png` | Scatter plot lat/lon coloreado por precio |
| `03_tipo_habitacion.png` | Conteo y boxplot por tipo de habitacion |
| `04a_grafo_geografico.png` | Visualizacion del grafo KNN (submuestra) |
| `04b_grafo_textual.png` | Visualizacion del grafo textual (submuestra) |
| `05_predicciones.png` | Real vs predicho por modelo |
| `06_comparacion_metricas.png` | Barplot comparativo MSE/MAE/R2 |
| `07_learning_curves.png` | Curvas de aprendizaje GCN |
| `08_resumen_resultados.csv` | Tabla resumen en CSV |

---

## 5. Preguntas de Reflexion

### Por que usar grafos para datos inmobiliarios?

Los datos inmobiliarios tienen una estructura relacional natural: las propiedades
no son independientes, sino que sus precios se influencian mutuamente por
proximidad geografica, similitud de amenidades y competencia en el mercado. Un
grafo codifica estas relaciones explicitamente, permitiendo que el modelo
"pregunte a los vecinos" antes de hacer una prediccion. Esto es especialmente
valioso cuando hay pocas features individuales pero la relacion entre entidades
es informativa.

### Cual estrategia de grafo funciona mejor y por que?

El grafo textual (R2=0.40) supera al geografico (R2=0.24). Esto se debe a que:

1. **El texto captura el segmento de mercado:** Listings con descripciones similares
   compiten por el mismo tipo de huesped y por lo tanto tienen precios comparables.
2. **La geografia es ruidosa:** En Santorini, un hotel de lujo y un hostal pueden
   estar a 200 metros de distancia pero tener precios 10x diferentes.
3. **El umbral de similitud controla la densidad:** Con umbral 0.80, el grafo
   textual es mas selectivo (18K aristas vs 55K del geografico), reduciendo
   el oversmoothing.

### Que limitaciones tiene el GCN comparado con modelos tabulares?

1. **Inductive bias inadecuado:** La GCN asume que vecinos en el grafo tienen
   labels similares (homofilia). Si el grafo no refleja bien esta propiedad,
   el message passing introduce ruido en lugar de informacion.
2. **Oversmoothing:** Con pocas capas la GCN apenas propaga informacion; con
   muchas capas las representaciones convergen y se pierde discriminacion.
3. **Escalabilidad:** El grafo completo debe cargarse en memoria, lo cual es
   prohibitivo para ciudades con millones de listings.
4. **No captura interacciones no lineales complejas** entre features individuales
   tan bien como modelos de gradient boosting (XGBoost, LightGBM).

### Como mejorar los resultados?

- **Modelo hibrido:** Concatenar la salida de la GCN con las features originales
  y alimentar un MLP o gradient boosting.
- **Grafo multimodal:** Combinar aristas geograficas y textuales en un solo grafo
  con pesos aprendibles por tipo de arista.
- **GAT (Graph Attention Network):** Usar atencion para ponderar la importancia
  de cada vecino dinamicamente.
- **Features adicionales:** Incorporar amenidades parseadas, estacionalidad, y
  distancia a puntos de interes (caldera, playa, puerto).
- **Graph sampling:** Usar mini-batch training con GraphSAGE o ClusterGCN para
  mejorar generalizacion.

---

## 6. Estructura del Proyecto

```
Reto_III/
|-- main.py                 # Punto de entrada
|-- download_data.py        # Descarga del dataset (ejecutar una sola vez)
|-- datacard.md             # Data Card del dataset
|-- analisis.md             # Este archivo
|-- Reto_III_ML_AI.pdf      # Enunciado del reto
|-- data/
|   +-- listings.csv        # Datos de Airbnb Santorini
|-- src/
|   |-- __init__.py
|   |-- config.py           # Configuracion global
|   |-- data_loader.py      # Carga y preprocesamiento
|   |-- graph_builder.py    # Construccion de grafos (geo + texto)
|   |-- model.py            # Arquitectura GCN
|   |-- training.py         # Entrenamiento y evaluacion
|   +-- visualization.py    # Generacion de graficas
+-- resultados/             # Graficas y CSV generados
```

---

*Elaborado por Ana Lorena Jimenez Preciado -- Curso ML e AI, Colegio de Matematicas Bourbaki*
