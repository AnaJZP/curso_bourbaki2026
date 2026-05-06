# Data Card -- Airbnb Santorini Listings

## Informacion General

| Campo | Valor |
|---|---|
| **Nombre** | Airbnb Listings -- Santorini, Grecia |
| **Publicado por** | Inside Airbnb |
| **Anio** | 2022 (snapshot utilizado) |
| **Referencia** | [Inside Airbnb](http://insideairbnb.com/) |
| **Licencia** | Creative Commons CC0 1.0 |
| **Fuente utilizada** | [nkanak/predicting-prices-of-airbnb-listings](https://github.com/nkanak/predicting-prices-of-airbnb-listings) |
| **Tamano** | 4,540 listings, 54 columnas |

## Proposito

El dataset contiene informacion publica de listings de Airbnb en la isla de
Santorini, Grecia. Se utiliza para predecir precios de alojamiento mediante
Graph Neural Networks, explorando como la estructura de relaciones entre
propiedades (geograficas y textuales) puede mejorar las predicciones.

## Composicion

### Archivos

| Archivo | Shape | Descripcion |
|---|---|---|
| `listings.csv` | 4,540 x 54 | Datos completos de listings |

### Columnas principales

| Columna | Tipo | Descripcion |
|---|---|---|
| `id` | int | Identificador unico del listing |
| `name` | str | Titulo del listing |
| `description` | str | Descripcion completa |
| `price` | int | Precio por noche en USD |
| `latitude` | float | Latitud geografica |
| `longitude` | float | Longitud geografica |
| `room_type` | str | Tipo de habitacion (Entire home/apt, Private room, etc.) |
| `accommodates` | int | Numero de huespedes |
| `bedrooms` | float | Numero de dormitorios |
| `beds` | float | Numero de camas |
| `bathrooms_text` | str | Descripcion de banos |
| `review_scores_rating` | float | Puntuacion general (0-5) |
| `amenities` | str | Lista de amenidades en formato JSON |
| `availability_365` | int | Dias disponibles en un ano |

### Estadisticas clave

- **4,540 listings** en Santorini
- **Precio:** media $358, mediana $180, rango $9-$20,968
- **Room types:** Entire home (3,122), Private room (1,104), Hotel room (311), Shared (3)
- **Valores faltantes:** bedrooms 5%, beds 1%, review scores ~26%
- **Geolocalizacion:** 100% de listings tienen lat/lon

## Consideraciones

### Valores faltantes

- `bathrooms`: 100% NaN (se extrae de `bathrooms_text`)
- `review_scores_*`: ~26% NaN (listings sin reviews suficientes)
- `bedrooms`, `beds`: <5% NaN

### Sesgos conocidos

1. **Sesgo estacional:** Los precios capturados corresponden a un snapshot
   particular y pueden no reflejar la variacion estacional extrema de Santorini
   (alta temporada junio-septiembre vs baja temporada).
2. **Sesgo de seleccion:** Solo incluye listings activos en la plataforma
   al momento del scraping.
3. **Sesgo geografico:** Santorini tiene una concentracion extrema de listings
   de lujo (zonas de caldera) que inflan la media de precios.

### Usos apropiados

- Investigacion academica en prediccion de precios inmobiliarios
- Estudio de Graph Neural Networks aplicadas a datos espaciales
- Analisis exploratorio del mercado de alquiler vacacional

### Usos no apropiados

- Fijacion de precios comerciales sin datos actualizados
- Discriminacion de huespedes basada en perfiles

## Preprocesamiento Aplicado

1. Filtrado de listings con precio <= 0.
2. Eliminacion de outliers (percentil 99 de precio).
3. Extraccion de numero de banos desde `bathrooms_text`.
4. One-hot encoding de `room_type`.
5. Imputacion de NaN con mediana por columna.
6. Estandarizacion de features numericas (media 0, std 1).
7. Transformacion logaritmica del precio: log(precio + 1).

## Cita

```
Inside Airbnb. (2022). Santorini, Greece -- Airbnb Listings Data.
http://insideairbnb.com/
```
