# Curso Bourbaki 2026 -- Machine Learning e Inteligencia Artificial

Repositorio de retos del curso ML e IA del Colegio de Matematicas Bourbaki.

**Autora:** Ana Lorena Jimenez Preciado

## Estructura

```
curso_bourbaki2026/
|-- Reto_I/     Redes Multi-Modales: Deteccion de Memes Ofensivos
|-- Reto_II/    Series Temporales: Prediccion de Produccion Industrial
|-- Reto_III/   (proximamente)
|-- Reto_IV/    (proximamente)
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

## Dependencias

- Python 3.10+
- PyTorch
- Transformers (HuggingFace) -- solo Reto I
- scikit-learn
- matplotlib, seaborn
- pandas, numpy
- Pillow, tqdm -- solo Reto I
