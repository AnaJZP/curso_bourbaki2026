# Data Card -- Dynamic Pricing Dataset

## Informacion General

| Campo | Valor |
|---|---|
| **Nombre** | Dynamic Pricing Dataset |
| **Publicado por** | Mobius (Kaggle) |
| **Anio** | 2024 |
| **Referencia** | [Kaggle](https://www.kaggle.com/datasets/arashnic/dynamic-pricing-dataset) |
| **Licencia** | CC0: Public Domain |
| **Tamano** | 1,000 registros, 10 columnas |

## Proposito

El dataset simula datos historicos de una empresa de ride-sharing que busca
implementar una estrategia de precios dinamicos. Actualmente la empresa solo
usa la duracion del viaje para fijar tarifas. El objetivo es desarrollar un
modelo que ajuste precios en tiempo real considerando demanda, oferta,
ubicacion, tipo de vehiculo y otros factores del mercado.

## Composicion

### Archivos

| Archivo | Shape | Descripcion |
|---|---|---|
| `dynamic_pricing.csv` | 1,000 x 10 | Datos historicos de viajes |

### Columnas

| Columna | Tipo | Descripcion |
|---|---|---|
| `Number_of_Riders` | int | Numero de pasajeros buscando viaje (demanda) |
| `Number_of_Drivers` | int | Numero de conductores disponibles (oferta) |
| `Location_Category` | str | Tipo de ubicacion: Rural, Suburban, Urban |
| `Customer_Loyalty_Status` | str | Nivel de lealtad: Regular, Silver, Gold |
| `Number_of_Past_Rides` | int | Viajes previos del cliente |
| `Average_Ratings` | float | Calificacion promedio del conductor |
| `Time_of_Booking` | str | Momento del dia: Morning, Afternoon, Evening, Night |
| `Vehicle_Type` | str | Tipo de vehiculo: Economy, Premium |
| `Expected_Ride_Duration` | int | Duracion esperada del viaje (minutos) |
| `Historical_Cost_of_Ride` | float | Costo historico del viaje (USD) |

### Estadisticas clave

- **1,000 registros** de viajes simulados
- **Precio historico:** media $372.50, rango $25.99 - $836.12
- **Riders:** rango 20-100
- **Drivers:** rango 5-89
- **Ubicaciones:** Urban (346), Rural (332), Suburban (322)
- **Vehiculos:** Premium (522), Economy (478)
- **Sin valores faltantes**

## Consideraciones

### Datos simulados

Este dataset es sintetico/simulado, no proviene de una empresa real de
ride-sharing. Los patrones de demanda y precios son representativos pero
no reflejan una situacion de mercado real.

### Usos apropiados

- Aprendizaje de tecnicas de Reinforcement Learning
- Estudio del problema de Dynamic Pricing
- Experimentacion con DQN y variantes

### Usos no apropiados

- Fijacion de precios comerciales reales
- Analisis de mercado de transporte

## Preprocesamiento Aplicado

1. Calculo de ratio demanda/oferta (Riders / Drivers).
2. Encoding ordinal de variables categoricas.
3. Estandarizacion (media 0, std 1) de features numericas.
4. Historical_Cost_of_Ride se usa como precio base de referencia.

## Cita

```
Mobius (arashnic). (2024). Dynamic Pricing Dataset.
https://www.kaggle.com/datasets/arashnic/dynamic-pricing-dataset
```
