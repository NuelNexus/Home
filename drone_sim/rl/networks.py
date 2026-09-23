"""Neural networks for the flight policy."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def mlp(sizes, act=nn.Tanh, out_gain=1.0):
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        last = i == len(sizes) - 2
        nn.init.orthogonal_(lin.weight, out_gain if last else np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if not last:
            layers.append(act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Gaussian policy (actor) and state-value function (critic).

    The actor is deliberately small (2 x 64 tanh) so it can run on a
    microcontroller next to a real MPU-6050.
    """

    def __init__(self, obs_dim: int, act_dim: int, actor_hidden=(64, 64), critic_hidden=(256, 256),
                 init_log_std: float = -0.7):
        super().__init__()
        self.actor_hidden, self.critic_hidden = list(actor_hidden), list(critic_hidden)
        self.actor = mlp([obs_dim, *actor_hidden, act_dim], out_gain=0.01)
        self.critic = mlp([obs_dim, *critic_hidden, 1], out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def dist(self, obs):
        mu = self.actor(obs)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


class RunningMeanStd:
    """Streaming observation normaliser (parallel Welford update)."""

    def __init__(self, shape, clip: float = 10.0):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta**2 * self.count * bc / tot
        self.var, self.count = m2 / tot, tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count, "clip": self.clip}

    def load_state_dict(self, d):
        self.mean, self.var = np.asarray(d["mean"]), np.asarray(d["var"])
        self.count, self.clip = d["count"], d.get("clip", 10.0)
