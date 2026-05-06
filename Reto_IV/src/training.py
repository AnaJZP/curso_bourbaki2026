"""
training.py -- Entrenamiento y evaluacion del Reto IV.
"""

from dataclasses import dataclass, field

import numpy as np

from .config import (
    DQN_EPISODES,
    DQN_STEPS_PER_EP,
    DQN_TARGET_UPDATE,
    PRICE_MULTIPLIERS,
)
from .environment import DynamicPricingEnv
from .model import DQNAgent


@dataclass
class PolicyResult:
    """Contenedor de resultados por politica."""
    name: str
    total_revenue: float = 0.0
    avg_revenue_per_ride: float = 0.0
    avg_multiplier: float = 0.0
    avg_acceptance: float = 0.0
    multipliers_used: list = field(default_factory=list)
    revenues: list = field(default_factory=list)


# -- Entrenamiento DQN -------------------------------------------

def train_dqn(env: DynamicPricingEnv,
              episodes: int = DQN_EPISODES) -> tuple[DQNAgent, dict]:
    """
    Entrena el agente DQN.

    Retorna:
        agent: agente entrenado
        history: diccionario con metricas por episodio
    """
    state_dim = env.state_dim
    agent = DQNAgent(state_dim)

    history = {
        "episode_rewards": [],
        "episode_losses": [],
        "epsilons": [],
    }

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0.0
        ep_loss = 0.0
        steps = 0

        for step in range(DQN_STEPS_PER_EP):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store(state, action, reward, next_state, done)
            loss = agent.learn()

            state = next_state
            ep_reward += reward
            ep_loss += loss
            steps += 1

        # Actualizar target network
        if (ep + 1) % DQN_TARGET_UPDATE == 0:
            agent.update_target()

        agent.decay_epsilon()

        avg_loss = ep_loss / max(steps, 1)
        history["episode_rewards"].append(ep_reward)
        history["episode_losses"].append(avg_loss)
        history["epsilons"].append(agent.epsilon)

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(history["episode_rewards"][-50:])
            print(f"  Ep {ep + 1:4d}/{episodes} | "
                  f"Reward: {avg_r:8.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Eps: {agent.epsilon:.3f}")

    return agent, history


# -- Evaluacion de politicas ------------------------------------

def evaluate_dqn(agent: DQNAgent, env: DynamicPricingEnv) -> PolicyResult:
    """Evalua la politica DQN sobre todo el dataset."""
    result = PolicyResult(name="dqn")

    for idx in range(env.n_samples):
        state, base_price, demand_ratio = env.get_state_at(idx)
        action = agent.select_action(state, greedy=True)
        multiplier = PRICE_MULTIPLIERS[action]

        acceptance = env._acceptance_probability(multiplier, demand_ratio)
        revenue = base_price * multiplier * acceptance

        result.multipliers_used.append(multiplier)
        result.revenues.append(revenue)

    result.total_revenue = sum(result.revenues)
    result.avg_revenue_per_ride = np.mean(result.revenues)
    result.avg_multiplier = np.mean(result.multipliers_used)
    result.avg_acceptance = np.mean([
        env._acceptance_probability(m, env.demand_ratios[i])
        for i, m in enumerate(result.multipliers_used)
    ])

    return result


def evaluate_fixed(env: DynamicPricingEnv,
                   multiplier: float = 1.0) -> PolicyResult:
    """Evalua una politica de precio fijo (multiplicador constante)."""
    result = PolicyResult(name="fixed")

    for idx in range(env.n_samples):
        _, base_price, demand_ratio = env.get_state_at(idx)
        acceptance = env._acceptance_probability(multiplier, demand_ratio)
        revenue = base_price * multiplier * acceptance

        result.multipliers_used.append(multiplier)
        result.revenues.append(revenue)

    result.total_revenue = sum(result.revenues)
    result.avg_revenue_per_ride = np.mean(result.revenues)
    result.avg_multiplier = multiplier
    result.avg_acceptance = np.mean([
        env._acceptance_probability(multiplier, env.demand_ratios[i])
        for i in range(env.n_samples)
    ])

    return result


def evaluate_proportional(env: DynamicPricingEnv) -> PolicyResult:
    """
    Evalua una politica proporcional: multiplicador = f(demand_ratio).
    Sube precio cuando demanda > oferta, baja cuando hay exceso de oferta.
    """
    result = PolicyResult(name="proportional")

    for idx in range(env.n_samples):
        _, base_price, demand_ratio = env.get_state_at(idx)

        # Multiplicador proporcional al ratio demanda/oferta
        # Clipear entre 0.7 y 1.5
        multiplier = float(np.clip(demand_ratio, 0.7, 1.5))

        acceptance = env._acceptance_probability(multiplier, demand_ratio)
        revenue = base_price * multiplier * acceptance

        result.multipliers_used.append(multiplier)
        result.revenues.append(revenue)

    result.total_revenue = sum(result.revenues)
    result.avg_revenue_per_ride = np.mean(result.revenues)
    result.avg_multiplier = np.mean(result.multipliers_used)
    result.avg_acceptance = np.mean([
        env._acceptance_probability(result.multipliers_used[i], env.demand_ratios[i])
        for i in range(env.n_samples)
    ])

    return result


# -- Orquestador -------------------------------------------------

def run_all_models(env: DynamicPricingEnv) -> tuple[dict, DQNAgent, dict]:
    """
    Entrena DQN y evalua todas las politicas.

    Retorna:
        results: dict de PolicyResult por politica
        agent: agente DQN entrenado
        history: historial de entrenamiento
    """
    print("\n  --- Entrenando DQN ---")
    agent, history = train_dqn(env)

    results = {}

    print("\n  --- Evaluando politicas ---")

    print("  Politica: Precio Fijo (x1.0)")
    results["fixed"] = evaluate_fixed(env, multiplier=1.0)
    print(f"    Revenue total: ${results['fixed'].total_revenue:,.2f}")
    print(f"    Revenue medio: ${results['fixed'].avg_revenue_per_ride:.2f}")

    print("  Politica: Proporcional a demanda")
    results["proportional"] = evaluate_proportional(env)
    print(f"    Revenue total: ${results['proportional'].total_revenue:,.2f}")
    print(f"    Revenue medio: ${results['proportional'].avg_revenue_per_ride:.2f}")

    print("  Politica: DQN")
    results["dqn"] = evaluate_dqn(agent, env)
    print(f"    Revenue total: ${results['dqn'].total_revenue:,.2f}")
    print(f"    Revenue medio: ${results['dqn'].avg_revenue_per_ride:.2f}")

    return results, agent, history
