# Curso Bourbaki 2026 -- Machine Learning e Inteligencia Artificial

Repositorio de retos del curso ML e IA del Colegio de Matematicas Bourbaki.

**Autora:** Ana Lorena Jimenez Preciado

## Estructura

```
curso_bourbaki2026/
|-- Reto_I/     Redes Multi-Modales: Deteccion de Memes Ofensivos
|-- Reto_II/    Series Temporales: Prediccion de Produccion Industrial
|-- Reto_III/   Graph Neural Networks: Prediccion de Precios de Airbnb
|-- Reto_IV/    Deep Reinforcement Learning: Dynamic Pricing
```

---

## Reto I -- Redes Multi-Modales

Clasificador multi-modal (imagen + texto) para detectar memes ofensivos utilizando
embeddings de CLIP (ViT-B/32) y tres estrategias de fusion: concatenacion, producto
de Hadamard y combinada.

| Estrategia | AUROC | Accuracy |
|---|---|---|
| CONCAT | 0.743 | 71.2% |
| HADAMARD | 0.601 | 64.1% |
| COMBINED | 0.748 | 73.2% |

Para mas detalles, consultar `Reto_I/analisis.md`.

---

## Reto II -- Series Temporales

Prediccion de produccion semanal de activos industriales a partir de mediciones
diarias de sensores. Tres enfoques comparados: deteccion de anomalias (baseline),
GradientBoosting y LSTM.

| Modelo | Grupo 2 (RMSE) | Grupo 3 (RMSE) |
|---|---|---|
| BASELINE | 184,936 | 747,381 |
| GBR | 255,801 | 485,510 |
| LSTM | 389,029 | 523,797 |

Para mas detalles, consultar `Reto_II/analisis.md`.

---

## Reto III -- Graph Neural Networks

Prediccion de precios de Airbnb en Santorini (Grecia) mediante Graph Neural
Networks. Se construyen grafos por proximidad geografica (KNN) y similitud
textual (sentence-transformers) y se comparan contra regresion lineal.

| Modelo | MSE | RMSE | MAE | R2 |
|---|---|---|---|---|
| LINEAR | 56,806 | 238 | 127 | 0.5553 |
| GCN_GEO | 97,389 | 312 | 160 | 0.2376 |
| GCN_TXT | 76,152 | 276 | 137 | 0.4038 |

Para mas detalles, consultar `Reto_III/analisis.md`.

---

## Reto IV -- Deep Reinforcement Learning

Dynamic Pricing para ride-sharing mediante Deep Q-Learning. El agente aprende
a fijar precios optimos en funcion de demanda, oferta, ubicacion y tipo de vehiculo.

| Politica | Revenue Total | Revenue Medio | Multiplicador | Aceptacion |
|---|---|---|---|---|
| FIXED | $322,160 | $322.16 | 1.00 | 86.38% |
| PROPORTIONAL | $395,470 | $395.47 | 1.47 | 71.85% |
| DQN | $403,441 | $403.44 | 1.30 | 81.14% |

Para mas detalles, consultar `Reto_IV/analisis.md`.

## Como ejecutar

### Reto I

```bash
cd Reto_I
pip install torch transformers scikit-learn matplotlib seaborn pandas tqdm pillow
python download_data.py   # una sola vez
python main.py
```

### Reto II

```bash
cd Reto_II
pip install torch scikit-learn matplotlib seaborn pandas numpy
python main.py
```

### Reto III

```bash
cd Reto_III
pip install torch torch-geometric sentence-transformers scikit-learn matplotlib seaborn pandas
python download_data.py   # una sola vez
python main.py
```

### Reto IV

```bash
cd Reto_IV
pip install torch kagglehub scikit-learn matplotlib pandas
python download_data.py   # una sola vez
python main.py
```

## Dependencias

- Python 3.10+
- PyTorch
- PyTorch Geometric -- solo Reto III
- Transformers (HuggingFace) -- solo Reto I
- sentence-transformers -- solo Reto III
- kagglehub -- solo Reto IV (descarga de datos)
- scikit-learn
- matplotlib, seaborn
- pandas, numpy
- Pillow, tqdm -- solo Reto I
