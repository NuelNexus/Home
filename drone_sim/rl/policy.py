"""Deployment wrapper: load a trained checkpoint and fly with it."""
from __future__ import annotations

import numpy as np
import torch

from ..components import DroneModel
from ..env import ACT_DIM, OBS_DIM, EnvConfig
from .networks import ActorCritic, RunningMeanStd


class Policy:
    def __init__(self, net: ActorCritic, obs_rms: RunningMeanStd, env_config: dict | None = None,
                 drone_config: dict | None = None):
        self.net = net.eval()
        self.obs_rms = obs_rms
        self.env_config = env_config or {}
        self.drone_config = drone_config

    @classmethod
    def load(cls, path) -> "Policy":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = ActorCritic(OBS_DIM, ACT_DIM, ck["actor_hidden"], ck["critic_hidden"])
        net.load_state_dict(ck["model"])
        rms = RunningMeanStd(OBS_DIM)
        rms.load_state_dict(ck["obs_rms"])
        return cls(net, rms, ck.get("env_config"), ck.get("drone_config"))

    def env_cfg(self, **overrides) -> EnvConfig:
        d = dict(self.env_config)
        d.update(overrides)
        return EnvConfig.from_dict(d)

    def drone_model(self) -> DroneModel:
        return DroneModel(self.drone_config) if self.drone_config else DroneModel.from_json()

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(self.obs_rms.normalize(obs))
        return self.net.actor(x).clamp(-1, 1).numpy()
