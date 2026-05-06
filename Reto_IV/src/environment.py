"""
environment.py -- Entorno MDP para Dynamic Pricing (Reto IV).

Modela el problema de fijacion de precios como un Proceso de Decision
de Markov (MDP) donde:
    - Estado:    vector de features del mercado
    - Accion:    multiplicador de precio (discreto)
    - Recompensa: revenue estimado = precio * prob_aceptacion
    - Transicion: muestreo aleatorio del siguiente estado del dataset
"""

import numpy as np

from .config import N_ACTIONS, PRICE_MULTIPLIERS, SEED


class DynamicPricingEnv:
    """
    Entorno de Dynamic Pricing para Reinforcement Learning.

    El agente observa el estado del mercado (demanda, oferta, ubicacion, etc.)
    y elige un multiplicador de precio. La recompensa es el revenue esperado,
    que depende del precio fijado y la probabilidad de que el cliente acepte.
    """

    def __init__(self, states: np.ndarray, base_prices: np.ndarray,
                 demand_ratios: np.ndarray):
        """
        Args:
            states: matriz de estados (N, state_dim) normalizados
            base_prices: precios base historicos (N,)
            demand_ratios: ratio demanda/oferta crudo (N,)
        """
        self.states = states
        self.base_prices = base_prices
        self.demand_ratios = demand_ratios
        self.n_samples = len(states)
        self.state_dim = states.shape[1]
        self.n_actions = N_ACTIONS

        self.rng = np.random.RandomState(SEED)
        self.current_idx = 0

    def reset(self) -> np.ndarray:
        """Reinicia el entorno y devuelve un estado inicial aleatorio."""
        self.current_idx = self.rng.randint(0, self.n_samples)
        return self.states[self.current_idx].copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Ejecuta una accion y devuelve (next_state, reward, done, info).

        Args:
            action: indice del multiplicador de precio

        Returns:
            next_state: siguiente estado
            reward: revenue obtenido
            done: siempre False (entorno continuo, se controla externamente)
            info: diccionario con detalles
        """
        multiplier = PRICE_MULTIPLIERS[action]
        base_price = self.base_prices[self.current_idx]
        demand_ratio = self.demand_ratios[self.current_idx]

        # Precio final
        final_price = base_price * multiplier

        # Probabilidad de aceptacion del cliente
        # - Mas alta cuando el multiplicador es bajo
        # - Mas alta cuando la demanda relativa a oferta es alta
        # - Sigmoide parametrizada
        acceptance_prob = self._acceptance_probability(multiplier, demand_ratio)

        # Revenue = precio * probabilidad de aceptacion
        reward = final_price * acceptance_prob

        # Transicion: muestrear siguiente estado aleatorio
        self.current_idx = self.rng.randint(0, self.n_samples)
        next_state = self.states[self.current_idx].copy()

        info = {
            "multiplier": multiplier,
            "base_price": base_price,
            "final_price": final_price,
            "acceptance_prob": acceptance_prob,
            "demand_ratio": demand_ratio,
        }

        return next_state, reward, False, info

    def _acceptance_probability(self, multiplier: float,
                                demand_ratio: float) -> float:
        """
        Calcula la probabilidad de que el cliente acepte el precio.

        Modelo:
            prob = sigmoid(alpha * (demand_ratio - beta) - gamma * (multiplier - 1))

        Donde:
            - Alta demanda (demand_ratio > 1) aumenta la probabilidad
            - Multiplicador alto (>1) disminuye la probabilidad
            - El equilibrio esta en multiplier=1.0 con demand_ratio=1.0
        """
        alpha = 2.0   # sensibilidad a la demanda
        beta = 1.0    # punto de equilibrio de demanda
        gamma = 3.0   # sensibilidad al precio

        z = alpha * (demand_ratio - beta) - gamma * (multiplier - 1.0)
        prob = 1.0 / (1.0 + np.exp(-z))

        # Clipear para evitar 0 o 1 exactos
        return float(np.clip(prob, 0.05, 0.95))

    def get_state_at(self, idx: int) -> tuple[np.ndarray, float, float]:
        """Devuelve estado, precio base y demand ratio para un indice dado."""
        return (
            self.states[idx].copy(),
            self.base_prices[idx],
            self.demand_ratios[idx],
        )
