"""
model.py -- Modelos para el Reto IV.

Componentes:
1. DQNetwork: red neuronal que estima Q-values para cada accion
2. ReplayBuffer: buffer circular de experiencias
3. DQNAgent: agente con epsilon-greedy, target network y experience replay
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import (
    DEVICE,
    DQN_BATCH_SIZE,
    DQN_BUFFER_SIZE,
    DQN_EPS_DECAY,
    DQN_EPS_END,
    DQN_EPS_START,
    DQN_GAMMA,
    DQN_HIDDEN,
    DQN_LR,
    N_ACTIONS,
    SEED,
)


class DQNetwork(nn.Module):
    """
    Red neuronal para estimar Q(s, a) para todas las acciones.

    Arquitectura:
        Linear(state_dim, hidden) -> ReLU
        Linear(hidden, hidden) -> ReLU
        Linear(hidden, n_actions)
    """

    def __init__(self, state_dim: int, n_actions: int = N_ACTIONS,
                 hidden: int = DQN_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Buffer circular de experiencias para experience replay."""

    def __init__(self, capacity: int = DQN_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int = DQN_BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Agente Deep Q-Learning con:
    - Epsilon-greedy exploration
    - Experience replay
    - Target network (actualizacion periodica)
    """

    def __init__(self, state_dim: int, n_actions: int = N_ACTIONS):
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Redes
        self.policy_net = DQNetwork(state_dim, n_actions).to(DEVICE)
        self.target_net = DQNetwork(state_dim, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=DQN_LR)
        self.criterion = nn.MSELoss()

        # Buffer y epsilon
        self.buffer = ReplayBuffer()
        self.epsilon = DQN_EPS_START
        self.gamma = DQN_GAMMA

        random.seed(SEED)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Selecciona accion con epsilon-greedy."""
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        """Almacena experiencia en el buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float:
        """Realiza un paso de aprendizaje. Retorna la loss."""
        if len(self.buffer) < DQN_BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample()

        states_t = torch.tensor(states).to(DEVICE)
        actions_t = torch.tensor(actions).unsqueeze(1).to(DEVICE)
        rewards_t = torch.tensor(rewards).to(DEVICE)
        next_states_t = torch.tensor(next_states).to(DEVICE)
        dones_t = torch.tensor(dones).to(DEVICE)

        # Q(s, a) actual
        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Q-target: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.criterion(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """Copia los pesos de policy_net a target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decae epsilon multiplicativamente."""
        self.epsilon = max(DQN_EPS_END, self.epsilon * DQN_EPS_DECAY)
