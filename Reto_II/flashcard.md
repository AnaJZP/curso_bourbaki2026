# Flash Card -- Industrial Asset Production (Kayrros / ENS Challenge)

## Informacion General

| Campo | Valor |
|---|---|
| **Nombre** | Kayrros Industrial Asset Production Challenge |
| **Publicado por** | Kayrros / ENS (Ecole Normale Superieure) |
| **Anio** | 2020 |
| **Referencia** | [Challenge Data ENS #39](https://challengedata.ens.fr/challenges/39) |
| **Licencia** | Uso academico |
| **Fuente utilizada** | [Bourbaki/Datos-del-reto-II](https://github.com/pedro9olivares/Bourbaki/tree/main/BBVA/ML-%26-AI/Datos-del-reto-II) |

## Proposito

Estimar la produccion semanal agregada de grupos de activos industriales
a partir de mediciones diarias de sensores. Las mediciones pueden indicar
incidentes, mantenimiento u otros eventos que afectan la productividad.

## Composicion

### Archivos

| Archivo | Shape | Descripcion |
|---|---|---|
| `X_train.csv` | 241,696 x 7 | Mediciones diarias en formato tidy |
| `y_train.csv` | 104 x 3 | Produccion semanal por grupo |
| `assets.csv` | 83 x 2 | Capacidad nominal por activo |

### Esquema de X_train

| Columna | Tipo | Descripcion |
|---|---|---|
| `SAMPLE_ID` | int | Identificador de semana (1-104) |
| `GROUP_ID` | int | Grupo del activo (2 o 3) |
| `ASSET_ID` | int | Identificador del activo (1-83) |
| `MEASURE_TYPE` | int | Tipo de medicion (1-4) |
| `MEASURE_VALUE` | float | Valor de la medicion (29.5% NaN) |
| `MEASURE_WEEKDAY` | int | Dia de la semana (1=jueves a 7=miercoles) |
| `MEASURE_WEEK` | int | Semana de la medicion |

### Esquema de y_train

| Columna | Tipo | Descripcion |
|---|---|---|
| `SAMPLE_ID` | int | Semana |
| `PRODUCTION_GROUP_2` | float | Produccion del grupo 2 |
| `PRODUCTION_GROUP_3` | float | Produccion del grupo 3 |

### Estadisticas clave

- **83 activos** divididos en 2 grupos
- **104 semanas** de datos
- **4 tipos de medicion** por activo por dia
- **Capacidad nominal:** rango 4,100 a 603,000 (media 167,159)
- **Produccion grupo 2:** media ~3.79M, rango 2.85M-4.14M
- **Produccion grupo 3:** media ~9.14M, rango 8.21M-9.77M

## Consideraciones

### Valores faltantes

29.5% de las mediciones son NaN. Algunos activos no tienen ciertos tipos
de medida (series completas de NaN). Esto no necesariamente indica falta
de datos, sino que cierto tipo de sensor no esta disponible para ese activo.

### Restricciones del dominio

- Un activo no puede producir mas del 120% de su capacidad nominal.
- La produccion se reporta de forma agregada por grupo y semanal,
  mientras las mediciones son individuales y diarias.
- Las mediciones pueden corresponder a incidentes o mantenimiento.

### Sesgos conocidos

- Los datos son anonimizados: no se conoce el tipo real de los activos
  ni la ubicacion geografica.
- La ausencia de variables exogenas (precios, demanda, clima) limita
  la capacidad predictiva del modelo.

## Cita

```
Kayrros. (2020). Industrial Asset Production Challenge.
Challenge Data, ENS Paris.
https://challengedata.ens.fr/challenges/39
```
