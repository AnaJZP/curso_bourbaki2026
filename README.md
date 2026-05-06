# Curso Bourbaki 2026 -- Machine Learning e Inteligencia Artificial

Repositorio de retos del curso ML e IA del Colegio de Matematicas Bourbaki.

**Autora:** Ana Lorena Jimenez Preciado

## Estructura

```
curso_bourbaki2026/
|-- Reto_I/     Redes Multi-Modales: Deteccion de Memes Ofensivos
|-- Reto_II/    (proximamente)
|-- Reto_III/   (proximamente)
|-- Reto_IV/    (proximamente)
```

## Reto I -- Redes Multi-Modales

Clasificador multi-modal (imagen + texto) para detectar memes ofensivos utilizando
embeddings de CLIP (ViT-B/32) y tres estrategias de fusion: concatenacion, producto
de Hadamard y combinada.

**Resultados destacados:**

| Estrategia | AUROC | Accuracy |
|---|---|---|
| CONCAT | 0.743 | 71.2% |
| HADAMARD | 0.601 | 64.1% |
| COMBINED | 0.748 | 73.2% |

Para mas detalles, consultar `Reto_I/analisis.md`.

### Como ejecutar

```bash
cd Reto_I

# 1. Instalar dependencias
pip install torch transformers scikit-learn matplotlib seaborn pandas tqdm pillow

# 2. Descargar el dataset (una sola vez)
python download_data.py

# 3. Ejecutar el pipeline completo
python main.py
```

## Dependencias

- Python 3.10+
- PyTorch
- Transformers (HuggingFace)
- scikit-learn
- matplotlib, seaborn
- pandas, numpy
- Pillow
- tqdm
