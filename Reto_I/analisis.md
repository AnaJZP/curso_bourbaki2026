# Reto I -- Analisis y Reflexiones
**Redes Multi-Modales: Deteccion de Memes Ofensivos**

**Autora:** Ana Lorena Jimenez Preciado
**Curso:** Machine Learning e Inteligencia Artificial -- Colegio de Matematicas Bourbaki

---

## 1. Resumen del Problema

El objetivo de este reto es construir un clasificador multi-modal (imagen + texto) para detectar
memes ofensivos, utilizando embeddings de CLIP (ViT-B/32) y tecnicas de fusion de caracteristicas.

El dataset proviene del Hateful Memes Challenge (Meta AI, NeurIPS 2020), que plantea la deteccion
de discurso de odio como un problema inherentemente multi-modal: el significado ofensivo frecuentemente
surge de la combinacion de texto e imagen, no de cada modalidad por separado.

---

## 2. Conceptos Clave

| Concepto | Definicion |
|---|---|
| **CLIP** | Modelo pre-entrenado de OpenAI que aprende representaciones conjuntas de imagenes y texto mediante aprendizaje contrastivo. |
| **Embeddings** | Representaciones vectoriales densas de datos en un espacio latente de dimension fija (512 para CLIP ViT-B/32). |
| **Fusion Multi-Modal** | Estrategia para combinar informacion de distintas modalidades (concatenacion, Hadamard, combinacion). |
| **Producto de Hadamard** | Multiplicacion elemento a elemento: z = x (circled dot) y. Captura interacciones multiplicativas entre modalidades. |
| **Aprendizaje Contrastivo** | Paradigma que acerca representaciones de pares positivos y aleja las de pares negativos en el espacio latente. |
| **Transfer Learning** | Reutilizacion de conocimiento de un modelo pre-entrenado para una tarea downstream especifica. |
| **AUROC** | Area bajo la curva ROC; metrica de capacidad discriminativa de un clasificador binario. |

---

## 3. Metodologia

### 3.1 Pipeline

```
Memes (img + text) --> CLIP --> Embeddings (512d) --> Fusion --> Clasificador --> Prediccion
```

### 3.2 Estrategias de Fusion

| Estrategia | Descripcion | Dimension |
|---|---|---|
| **Concatenacion** | Se concatenan ambos vectores | 1024 |
| **Hadamard** | Producto elemento a elemento | 512 |
| **Combinada** | Concatenacion + producto Hadamard | 1536 |

### 3.3 Arquitectura del Clasificador

Red feedforward de 3 capas:

```
Linear(input_dim -> 256) -> BatchNorm -> GELU -> Dropout(0.3)
Linear(256 -> 128) -> BatchNorm -> GELU -> Dropout(0.3)
Linear(128 -> 1)
```

- **Loss:** BCEWithLogitsLoss
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** CosineAnnealing (T_max=30)
- **Epocas:** 30

---

## 4. Resultados

Los resultados exactos dependen de la ejecucion. A continuacion se muestra un ejemplo representativo
obtenido durante el desarrollo:

| Estrategia | AUROC | Accuracy | Dim. Input | Best Val Loss |
|---|---|---|---|---|
| CONCAT | 0.6441 | 62.50% | 1024 | 0.6529 |
| **HADAMARD** | **0.7205** | 56.25% | 512 | 0.6383 |
| COMBINED | 0.6476 | 62.50% | 1536 | 0.6445 |

### Observaciones

1. **Hadamard obtiene el mejor AUROC**, indicando mayor capacidad discriminativa a pesar
   de usar la dimension mas baja (512). Esto sugiere que las interacciones multiplicativas entre
   modalidades son mas informativas que la simple concatenacion.

2. **Concat y Combined obtienen igual accuracy**, pero Hadamard tiene accuracy menor.
   La diferencia AUROC vs accuracy sugiere que Hadamard produce probabilidades mejor calibradas
   pero con un umbral de decision suboptimo en 0.5.

3. **Overfitting visible** en las curvas de aprendizaje: el loss de entrenamiento baja a valores
   cercanos a cero mientras el de validacion sube, lo cual es una senal clara de sobreajuste
   en un dataset pequeno.

### Graficas Generadas

Todas las graficas se encuentran en la carpeta `resultados/`:

| Archivo | Contenido |
|---|---|
| `01_muestras_dataset.png` | 10 memes de ejemplo (5 por clase) |
| `02_distribucion_clases.png` | Distribucion de clases |
| `03_longitud_texto.png` | Box plot de longitud de texto por clase |
| `04_curvas_aprendizaje.png` | Loss train/val por estrategia |
| `05_curvas_roc.png` | Curvas ROC comparativas |
| `06_matrices_confusion.png` | Matrices de confusion |
| `07_resumen_resultados.csv` | Tabla resumen en CSV |

---

## 5. Preguntas de Reflexion

### Por que la concatenacion puede funcionar mejor que el producto de Hadamard?

La concatenacion preserva toda la informacion de ambas modalidades sin forzar interaccion
especifica, delegando al clasificador downstream la tarea de aprender que combinaciones importan.
El producto de Hadamard, en cambio, restringe las interacciones a multiplicaciones elemento
a elemento, lo que puede perder relaciones entre dimensiones distintas. Sin embargo, en nuestro
experimento Hadamard supero en AUROC, probablemente porque con pocas muestras el espacio de
1024 dimensiones es demasiado grande y el clasificador no logra generalizarlo.

### Que limitaciones tiene CLIP para detectar sarcasmo visual?

CLIP fue entrenado con aprendizaje contrastivo sobre pares (imagen, caption) de internet. Sus
principales limitaciones para sarcasmo son:

1. **Entrenado para correspondencia, no incongruencia:** CLIP aprende a emparejar imagenes
   con textos que las describen, pero el sarcasmo y el odio frecuentemente dependen de la
   incongruencia entre texto e imagen.
2. **Conocimiento cultural limitado:** Muchos memes ofensivos requieren contexto cultural
   especifico (estereotipos, eventos historicos) que CLIP no necesariamente codifica.
3. **Embeddings separados:** CLIP produce embeddings independientes para texto e imagen;
   no modela interacciones cruzadas profundas (cross-attention) entre modalidades.

### Como afecta el desbalance de clases al rendimiento?

En este caso el dataset esta aproximadamente balanceado (~50/50), por lo que no es un factor
dominante. Sin embargo, en datasets reales de moderacion, el desbalance es severo (>95% benigno).
Alternativas para abordar esto:

- **Focal Loss:** Reduce el peso de muestras faciles y enfoca el aprendizaje en las dificiles.
- **Oversampling / SMOTE:** Generar muestras sinteticas de la clase minoritaria.
- **Class weights:** Ponderar la perdida inversamente proporcional a la frecuencia de clase.

### Que implicaciones eticas tiene un sistema automatico de moderacion?

| Tipo de error | Consecuencia |
|---|---|
| **Falso Positivo** (censurar contenido benigno) | Censura indebida, silenciamiento de voces legitimas, erosion de confianza |
| **Falso Negativo** (no detectar odio) | Exposicion a contenido danino, normalizacion de discurso de odio |

Un sistema de moderacion debe:
- Nunca operar sin supervision humana en decisiones de alto impacto.
- Ser transparente sobre sus limitaciones y tasas de error.
- Ser auditable para detectar sesgos contra grupos especificos.
- Incluir mecanismos de apelacion para decisiones automatizadas.

---

## 6. Desafio Adicional: Attention-Based Fusion

Como extension propuesta, se puede implementar un mecanismo de atencion que aprenda a ponderar
dinamicamente cada modalidad segun el ejemplo:

```python
class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, text_emb, image_emb):
        combined = torch.cat([text_emb, image_emb], dim=-1)
        weights = self.attention(combined)  # (batch, 2)
        return weights[:, 0:1] * text_emb + weights[:, 1:2] * image_emb
```

Ventaja: el modelo puede aprender que para ciertos memes el texto es mas informativo,
y para otros la imagen. Esto es especialmente util cuando la ofensividad depende de una sola
modalidad.

---

## 7. Estructura del Proyecto

```
Reto_I/
|-- main.py                 # Punto de entrada
|-- download_data.py        # Descarga del dataset (ejecutar una sola vez)
|-- flashcard.md             # Data Card del dataset
|-- analisis.md             # Este archivo
|-- Reto_I_ML_AI.pdf        # Enunciado del reto
|-- data/
|   |-- data.csv            # Metadatos del dataset
|   +-- images/             # Imagenes PNG
|-- src/
|   |-- __init__.py
|   |-- config.py           # Configuracion global
|   |-- data_loader.py      # Carga de datos y EDA
|   |-- embeddings.py       # Extraccion de embeddings CLIP
|   |-- fusion.py           # Estrategias de fusion
|   |-- model.py            # Arquitectura del clasificador
|   |-- training.py         # Loop de entrenamiento
|   +-- visualization.py    # Generacion de graficas
+-- resultados/             # Graficas y CSV generados
```

---

*Elaborado por Ana Lorena Jimenez Preciado -- Curso ML e AI, Colegio de Matematicas Bourbaki*
