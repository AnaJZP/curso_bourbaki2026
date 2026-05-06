# Data Card -- Hateful Memes (Meta AI / Facebook AI)

## Informacion General

| Campo | Valor |
|---|---|
| **Nombre** | Hateful Memes Challenge |
| **Publicado por** | Meta (Facebook AI Research) |
| **Anio** | 2020 |
| **Referencia** | [Kiela et al., 2020 -- NeurIPS](https://arxiv.org/abs/2005.04790) |
| **Licencia** | Uso academico e investigacion |
| **Fuente utilizada** | [HuggingFace: neuralcatcher/hateful_memes](https://huggingface.co/datasets/neuralcatcher/hateful_memes) |
| **Subconjunto usado** | Memes con imagen descargada exitosamente (subconjunto del dataset original de 8500) |

## Proposito

El dataset fue creado para avanzar en la deteccion automatica de discurso de odio multi-modal,
un problema que requiere comprension conjunta de texto e imagen. Los memes son particularmente
dificiles porque el significado ofensivo surge de la combinacion de ambas modalidades, no de cada
una por separado.

## Composicion

### Estructura de archivos

```
data/
├── data.csv          # Metadatos: id, original_id, text, label, img_path
└── images/           # Imagenes en formato PNG
    ├── 01845.png
    ├── 02146.png
    └── ...
```

### Esquema del CSV

| Columna | Tipo | Descripcion |
|---|---|---|
| `id` | int | Identificador secuencial (0-indexed) |
| `original_id` | str | ID original del dataset de Meta |
| `text` | str | Texto superpuesto en el meme |
| `label` | int | 0 = no ofensivo, 1 = ofensivo |
| `img_path` | str | Ruta al archivo de imagen |

### Distribucion de clases

El dataset original esta aproximadamente balanceado (~50/50 entre ofensivo y no ofensivo).

### Estadisticas de texto

- **Longitud promedio:** ~60 caracteres
- **Rango:** 4 a 300+ caracteres
- **Idioma:** Ingles

## Consideraciones Eticas

**Advertencia:** Este dataset contiene contenido explicitamente ofensivo, incluyendo
lenguaje de odio dirigido a grupos etnicos, religiosos, de genero y orientacion sexual.
Se utiliza exclusivamente con fines de investigacion academica.

### Sesgos conocidos

1. **Sesgo cultural:** Los memes reflejan el contexto sociopolitico estadounidense (~2019-2020).
2. **Sesgo de anotacion:** Las etiquetas fueron asignadas por anotadores humanos con posibles
   discrepancias en criterios de lo que constituye odio.
3. **Subrepresentacion:** El subconjunto utilizado es una fraccion del dataset original
   (8500 memes), lo que puede no capturar toda la diversidad.

### Usos apropiados

- Investigacion academica en NLP y vision computacional
- Desarrollo y evaluacion de modelos de moderacion de contenido
- Educacion sobre deteccion de discurso de odio multi-modal

### Usos no apropiados

- Generacion de contenido ofensivo
- Vigilancia o perfilado de individuos
- Sistemas de moderacion desplegados sin supervision humana

## Preprocesamiento Aplicado

1. Descarga de imagenes desde HuggingFace.
2. Filtrado de muestras cuyas imagenes no se descargaron correctamente.
3. Reconstruccion de rutas de imagen portables basadas en `original_id`.
4. No se aplico augmentacion ni normalizacion sobre imagenes en esta etapa.
5. No se aplico preprocesamiento de texto (se delega al tokenizer de CLIP).

## Cita

```bibtex
@inproceedings{kiela2020hateful,
  title={The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes},
  author={Kiela, Douwe and Firooz, Hamed and Mober, Aravind and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```
