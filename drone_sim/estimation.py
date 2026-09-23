"""Mahony nonlinear complementary filter (vectorised).

Fuses MPU-6050 gyro and accelerometer readings into an attitude estimate,
exactly as an embedded flight controller would:

    e      = â × v̂                     (â: normalised accel, v̂ = R̂ᵀ e_z)
    ω_corr = ω_gyro - b̂ + Kp e
    ḃ      = -Ki e
    q̂̇      = ½ q̂ ⊗ [0, ω_corr]

The accelerometer correction is faded out when |a| is far from 1 g, because
during aggressive manoeuvres it no longer points along gravity.
"""
from __future__ import annotations

import numpy as np

from .math3d import cross, quat_from_axis_angle, quat_mul, quat_normalize, quat_to_rotmat


class MahonyFilter:
    def __init__(self, num: int, kp: float = 1.0, ki: float = 0.05, g: float = 9.80665):
        self.n, self.kp, self.ki, self.g = num, kp, ki, g
        self.q = np.zeros((num, 4))
        self.q[:, 0] = 1.0
        self.bias = np.zeros((num, 3))

    def reset(self, idx, quat) -> None:
        self.q[idx] = quat
        self.bias[idx] = 0.0

    def update(self, gyro: np.ndarray, accel: np.ndarray, dt: float, mask=None) -> None:
        R = quat_to_rotmat(self.q)
        v = R[:, 2, :]                                   # R̂ᵀ e_z = third row of R̂
        an = np.linalg.norm(accel, axis=1, keepdims=True)
        a = accel / np.maximum(an, 1e-6)
        trust = np.clip(1.0 - np.abs(an / self.g - 1.0) / 0.3, 0.0, 1.0)
        e = cross(a, v) * trust
        if mask is not None:
            e = e * mask[:, None]
        self.bias -= self.ki * e * dt
        w = gyro - self.bias + self.kp * e
        dq = quat_from_axis_angle(w * dt)
        q_new = quat_normalize(quat_mul(self.q, dq))
        if mask is None:
            self.q = q_new
        else:
            self.q[mask] = q_new[mask]

    def rotation(self) -> np.ndarray:
        return quat_to_rotmat(self.q)
