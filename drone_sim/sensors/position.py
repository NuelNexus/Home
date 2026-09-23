"""Generic position/velocity sensor (GPS + barometer, optical flow, or mocap).

The MPU-6050 alone cannot observe position, so autonomous flight needs an
external position fix.  This models one with white noise, a slowly wandering
bias (random walk, like GPS drift) and a lower update rate (zero-order hold).
"""
from __future__ import annotations

import numpy as np


class PositionSensor:
    def __init__(self, num: int, rate_hz: float = 50.0, pos_noise=(0.02, 0.02, 0.02), vel_noise=0.05,
                 bias_walk=0.005, rng: np.random.Generator | None = None):
        self.n = num
        self.rate_hz = rate_hz
        self.pos_noise = np.asarray(pos_noise, float)
        self.vel_noise = float(vel_noise)
        self.bias_walk = float(bias_walk)
        self.rng = rng or np.random.default_rng()
        self.bias = np.zeros((num, 3))
        self.pos = np.zeros((num, 3))
        self.vel = np.zeros((num, 3))
        self._acc_t = np.zeros(num)

    def reset(self, idx, pos, vel):
        self.bias[idx] = 0.0
        self.pos[idx] = pos
        self.vel[idx] = vel
        self._acc_t[idx] = 0.0

    def update(self, dt: float, true_pos: np.ndarray, true_vel: np.ndarray) -> None:
        self.bias += self.rng.standard_normal(self.bias.shape) * self.bias_walk * np.sqrt(dt)
        self._acc_t += dt
        ready = self._acc_t >= 1.0 / self.rate_hz - 1e-9
        if ready.any():
            k = int(ready.sum())
            self.pos[ready] = true_pos[ready] + self.bias[ready] + self.rng.standard_normal((k, 3)) * self.pos_noise
            self.vel[ready] = true_vel[ready] + self.rng.standard_normal((k, 3)) * self.vel_noise
            self._acc_t[ready] = 0.0
