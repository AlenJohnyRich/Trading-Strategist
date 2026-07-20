"""
Reinforcement Learning agent (PPO) skeleton for trading / decision tasks.
This module provides a simple PPO implementation suitable for simulation-based
training. It is intentionally compact; production usage should rely on stable
implementations like Stable-Baselines3 or RLlib and distributed training.
"""
from __future__ import annotations

import math
import random
import logging
from typing import Tuple, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
except Exception:
    torch = None

logger = logging.getLogger('TradingAI.rl')


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.logstd = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logstd.exp()


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.v(x).squeeze(-1)


class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, lr: float = 3e-4, clip: float = 0.2, device: Optional[str] = None):
        if torch is None:
            raise RuntimeError('torch required for PPOAgent')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = PolicyNet(obs_dim, act_dim).to(self.device)
        self.value = ValueNet(obs_dim).to(self.device)
        self.opt_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.opt_value = optim.Adam(self.value.parameters(), lr=lr)
        self.clip = clip

    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
        mu, std = self.policy(obs_t)
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1).item()
        return a.cpu().numpy(), logp

    def update(self, trajectories: List[dict], epochs: int = 4, batch_size: int = 64):
        # trajectories: list of dicts containing obs, actions, rewards, logp, returns, advantages
        # Flatten
        obs = torch.tensor(np.concatenate([t['obs'] for t in trajectories]), dtype=torch.float32).to(self.device)
        acts = torch.tensor(np.concatenate([t['acts'] for t in trajectories]), dtype=torch.float32).to(self.device)
        old_logp = torch.tensor(np.concatenate([t['logp'] for t in trajectories]), dtype=torch.float32).to(self.device)
        advs = torch.tensor(np.concatenate([t['adv'] for t in trajectories]), dtype=torch.float32).to(self.device)
        returns = torch.tensor(np.concatenate([t['ret'] for t in trajectories]), dtype=torch.float32).to(self.device)

        dataset_size = obs.shape[0]
        for ep in range(epochs):
            idxs = torch.randperm(dataset_size)
            for i in range(0, dataset_size, batch_size):
                batch_idx = idxs[i:i+batch_size]
                b_obs, b_acts, b_old_logp, b_advs, b_ret = obs[batch_idx], acts[batch_idx], old_logp[batch_idx], advs[batch_idx], returns[batch_idx]
                mu, std = self.policy(b_obs)
                dist = torch.distributions.Normal(mu, std)
                logp = dist.log_prob(b_acts).sum(-1)
                ratio = torch.exp(logp - b_old_logp)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * b_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                self.opt_policy.zero_grad(); policy_loss.backward(); self.opt_policy.step()

                # value loss
                val = self.value(b_obs)
                value_loss = ((val - b_ret) ** 2).mean()
                self.opt_value.zero_grad(); value_loss.backward(); self.opt_value.step()

        return True


# Utility: compute advantages and returns (GAE)
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = [0]*T
    lastgaelam = 0
    for t in reversed(range(T)):
        delta = rewards[t] + (gamma * values[t+1] if t+1 < T else 0) - values[t]
        lastgaelam = delta + gamma * lam * lastgaelam
        adv[t] = lastgaelam
    returns = [adv[i] + values[i] for i in range(T)]
    return adv, returns
