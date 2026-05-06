# Reto IV -- Analisis y Reflexiones
**Deep Reinforcement Learning: Dynamic Pricing**

**Autora:** Ana Lorena Jimenez Preciado
**Curso:** Machine Learning e Inteligencia Artificial -- Colegio de Matematicas Bourbaki

---

## 1. Resumen del Problema

El objetivo es entrenar un agente de Deep Q-Learning (DQN) que aprenda a fijar
precios optimos para viajes de ride-sharing en tiempo real, considerando factores
como demanda de pasajeros, oferta de conductores, ubicacion, tipo de vehiculo y
horario.

El problema se formula como un Proceso de Decision de Markov (MDP) donde:
- El agente observa el estado del mercado.
- Elige un multiplicador de precio sobre el precio base.
- Recibe una recompensa proporcional al revenue (precio x probabilidad de aceptacion).
- El objetivo es maximizar el revenue acumulado.

---

## 2. Conceptos Clave

| Concepto | Definicion |
|---|---|
| **Reinforcement Learning (RL)** | Paradigma de aprendizaje donde un agente aprende a tomar decisiones secuenciales maximizando una recompensa acumulada. |
| **MDP** | Proceso de Decision de Markov: formalizacion de un problema de decision secuencial con estados, acciones, transiciones y recompensas. |
| **Q-Learning** | Algoritmo de RL que aprende la funcion Q(s,a), el valor esperado de tomar la accion a en el estado s y seguir la politica optima despues. |
| **DQN** | Deep Q-Network: aproxima Q(s,a) con una red neuronal en lugar de una tabla, permitiendo estados continuos. |
| **Experience Replay** | Tecnica donde las experiencias (s,a,r,s') se almacenan en un buffer y se muestrean aleatoriamente para entrenamiento, rompiendo correlaciones temporales. |
| **Target Network** | Segunda red neuronal cuyos pesos se actualizan periodicamente, estabilizando el entrenamiento al evitar que el target se mueva durante la optimizacion. |
| **Epsilon-greedy** | Estrategia de exploracion: con probabilidad epsilon se elige accion aleatoria, con (1-epsilon) se elige la accion con mayor Q-value. |
| **Dynamic Pricing** | Estrategia de precios que ajusta tarifas en tiempo real basandose en condiciones de mercado (demanda, oferta, competencia). |

---

## 3. Metodologia

### 3.1 Formulacion del MDP

| Componente | Implementacion |
|---|---|
| **Estado (s)** | Vector de 10 features: demand_supply_ratio, riders, drivers, past_rides, ratings, duration, location, loyalty, time, vehicle |
| **Acciones (a)** | 8 multiplicadores discretos: [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50] |
| **Recompensa (r)** | revenue = precio_base x multiplicador x prob_aceptacion |
| **Transicion** | Muestreo aleatorio del siguiente estado del dataset |

### 3.2 Modelo de Aceptacion

La probabilidad de que un cliente acepte el precio se modela con una sigmoide:

```
prob = sigmoid(alpha * (demand_ratio - 1) - gamma * (multiplier - 1))
```

Donde:
- alpha = 2.0 (sensibilidad a la demanda)
- gamma = 3.0 (sensibilidad al precio)
- Cuando demanda > oferta: mayor tolerancia a precios altos
- Cuando oferta > demanda: el cliente es mas sensible al precio

### 3.3 Arquitectura DQN

```
Linear(10 -> 128) -> ReLU
Linear(128 -> 128) -> ReLU
Linear(128 -> 8)       [Q-values para cada multiplicador]
```

Hiperparametros:
- Optimizer: Adam, lr=0.001
- Gamma (descuento): 0.95
- Epsilon: 1.0 -> 0.05 (decay 0.995 por episodio)
- Buffer: 10,000 experiencias
- Batch: 64
- Target update: cada 10 episodios
- Gradient clipping: norma maxima 1.0

### 3.4 Baselines

| Baseline | Estrategia |
|---|---|
| **Fijo (x1.0)** | Siempre cobra el precio historico, sin ajustes |
| **Proporcional** | Multiplicador = demand_ratio (clipado a [0.7, 1.5]) |

---

## 4. Resultados

### 4.1 Comparacion de Politicas

| Politica | Revenue Total | Revenue Medio | Multiplicador | Aceptacion |
|---|---|---|---|---|
| FIXED | $322,160 | $322.16 | 1.00 | 86.38% |
| PROPORTIONAL | $395,470 | $395.47 | 1.47 | 71.85% |
| **DQN** | **$403,441** | **$403.44** | **1.30** | **81.14%** |

### 4.2 Observaciones

1. **El DQN supera a ambos baselines.** Con $403,441 de revenue total, el DQN
   mejora un 25.2% sobre el precio fijo y un 2.0% sobre la politica proporcional.

2. **El DQN encuentra el equilibrio optimo.** La politica proporcional sube
   precios agresivamente (multiplicador medio 1.47) pero pierde clientes
   (aceptacion 71.85%). El DQN usa un multiplicador medio de 1.30 con una
   aceptacion del 81.14%, encontrando el sweet spot entre precio y volumen.

3. **El agente aprende a diferenciar contextos.** El heatmap de acciones
   muestra que el DQN aplica multiplicadores altos cuando la demanda es alta
   (demand_ratio > 1.5) y multiplicadores bajos cuando la oferta supera a la
   demanda, demostrando que aprendio la dinamica del mercado.

4. **Convergencia del entrenamiento.** La curva de recompensa muestra que el
   agente converge alrededor del episodio 300, pasando de recompensas de ~14,000
   a ~20,000 por episodio conforme explota lo aprendido.

### Graficas Generadas

| Archivo | Contenido |
|---|---|
| `01_eda_demanda_oferta.png` | Ratio demanda/oferta y scatter riders vs drivers |
| `02_eda_precios.png` | Distribucion de precios por vehiculo, ubicacion y horario |
| `03_reward_curve.png` | Curva de recompensa y loss durante entrenamiento |
| `04_epsilon_decay.png` | Decaimiento de epsilon (exploracion -> explotacion) |
| `05_acciones_por_estado.png` | Heatmap de acciones DQN vs demand_ratio |
| `06_comparacion_revenue.png` | Revenue total y medio por politica |
| `07_distribucion_multiplicadores.png` | Histograma de multiplicadores usados |
| `08_resumen_resultados.csv` | Tabla resumen en CSV |

---

## 5. Preguntas de Reflexion

### Por que Reinforcement Learning para Dynamic Pricing?

El pricing dinamico es un problema de decision secuencial bajo incertidumbre:
- No hay una funcion objetivo diferenciable (depende de la respuesta del cliente).
- Las acciones tienen efectos a largo plazo (un precio muy alto hoy puede espantar
  clientes futuros).
- El entorno cambia (demanda fluctua con horario, ubicacion, temporada).

RL es ideal porque aprende directamente de la interaccion con el entorno,
sin necesitar un modelo explicito de la demanda. El agente descubre la
politica optima por prueba y error.

### Que aporta el DQN sobre Q-Learning tabular?

Q-Learning tabular necesita una tabla Q(s,a) para cada combinacion estado-accion.
Con 10 features continuas, el espacio de estados es infinito y la tabla seria
impracticable. El DQN reemplaza la tabla con una red neuronal que generaliza:
puede estimar Q-values para estados nunca vistos, interpolando desde estados
similares del entrenamiento.

### Por que el DQN no mejora mas sobre la politica proporcional?

La politica proporcional ya captura la intuicion principal (subir cuando hay
demanda, bajar cuando no), que es esencialmente lo que el DQN aprende. La
mejora del DQN viene de:
- **Considerar todas las features**, no solo el ratio demanda/oferta
- **Optimizar el nivel exacto** del multiplicador (1.30 vs 1.47)
- **Balancear precio y aceptacion** de forma no lineal

Con mas datos y un entorno mas complejo (efectos temporales, competencia), la
ventaja del DQN seria mayor.

### Limitaciones del enfoque

1. **Entorno simulado:** La funcion de aceptacion es un modelo simplificado.
   En la realidad, la respuesta del cliente depende de muchos factores no
   observados (precio de la competencia, urgencia, clima).
2. **Transiciones i.i.d.:** El muestreo aleatorio del siguiente estado no
   captura la dinamica temporal real del mercado.
3. **Espacio de acciones discreto:** 8 multiplicadores limitan la granularidad.
   En produccion se usaria un espacio continuo (DDPG, SAC).
4. **Sin efecto de largo plazo:** El MDP episodico no captura como los precios
   de hoy afectan la demanda de manana.

---

## 6. Estructura del Proyecto

```
Reto_IV/
|-- main.py                 # Punto de entrada (7 pasos)
|-- download_data.py        # Descarga del dataset (ejecutar una sola vez)
|-- datacard.md             # Data Card del dataset
|-- analisis.md             # Este archivo
|-- Reto_IV_ML_AI.pdf       # Enunciado del reto
|-- data/
|   +-- dynamic_pricing.csv # Datos de Kaggle
|-- src/
|   |-- __init__.py
|   |-- config.py           # Configuracion global
|   |-- data_loader.py      # Carga y preprocesamiento
|   |-- environment.py      # Entorno MDP para pricing
|   |-- model.py            # DQN (red neuronal, replay buffer, agente)
|   |-- training.py         # Entrenamiento y evaluacion
|   +-- visualization.py    # Generacion de graficas
+-- resultados/             # Graficas y CSV generados
```

---

*Elaborado por Ana Lorena Jimenez Preciado -- Curso ML e AI, Colegio de Matematicas Bourbaki*
