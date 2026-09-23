"""Classical cascaded controller used as a baseline for the neural network.

position P -> velocity PI -> desired thrust vector -> geometric attitude
controller on SO(3) (Lee, Leok & McClamroch 2010) -> body-rate P -> motor mixer.

It reads exactly the same (noisy) sensors as the RL policy: Mahony attitude
from the simulated MPU-6050, MPU-6050 gyro, and the position sensor.  It only
knows the *nominal* configured mass/inertia, not the randomised true values.
"""
from __future__ import annotations

import numpy as np

from ..math3d import cross


class CascadedController:
    def __init__(self, env, kp_pos=1.2, kp_vel=2.5, ki_vel=0.6, kp_att=9.0, rate_tau=0.05, max_vel=2.5,
                 max_tilt_deg=40.0, yaw_setpoint=0.0):
        self.env = env
        mp = env.model.mass_properties()
        self.mass, self.I = mp.mass, mp.inertia
        self.g = env.g
        self.kp_pos, self.kp_vel, self.ki_vel = kp_pos, kp_vel, ki_vel
        self.kp_att, self.rate_tau = kp_att, rate_tau
        self.max_vel, self.max_tilt = max_vel, np.deg2rad(max_tilt_deg)
        self.yaw_sp = yaw_setpoint
        self.vel_int = np.zeros((env.n, 3))
        m = env.model
        pos = m.rotor_positions - mp.com
        # Mixer: [T_total, τx, τy, τz]^T = A @ T_rotors
        self.A = np.vstack((np.ones(m.num_rotors), pos[:, 1], -pos[:, 0], -m.spin_dirs * m.km / m.kf))
        self.A_inv = np.linalg.pinv(self.A)
        self.t_max = m.max_thrust_per_rotor

    def reset(self, idx=None):
        if idx is None:
            self.vel_int[:] = 0.0
        else:
            self.vel_int[idx] = 0.0

    def __call__(self, obs=None) -> np.ndarray:
        env = self.env
        dt = env.dt
        R = env.ahrs.rotation()
        omega = env.imu.read_gyro()
        pos, vel = env.pos_sensor.pos, env.pos_sensor.vel

        vel_sp = self.kp_pos * (env.target - pos)
        n = np.linalg.norm(vel_sp, axis=1, keepdims=True)
        vel_sp *= np.minimum(1.0, self.max_vel / np.maximum(n, 1e-9))
        ev = vel_sp - vel
        self.vel_int = np.clip(self.vel_int + ev * dt, -2, 2)
        acc = self.kp_vel * ev + self.ki_vel * self.vel_int
        acc[:, 2] += self.g
        # Limit tilt: horizontal acceleration relative to vertical.
        acc[:, 2] = np.maximum(acc[:, 2], 0.3 * self.g)
        h = np.linalg.norm(acc[:, :2], axis=1)
        lim = acc[:, 2] * np.tan(self.max_tilt)
        acc[:, :2] *= np.minimum(1.0, lim / np.maximum(h, 1e-9))[:, None]
        F = self.mass * acc

        b3 = F / np.linalg.norm(F, axis=1, keepdims=True)
        c1 = np.array([np.cos(self.yaw_sp), np.sin(self.yaw_sp), 0.0])
        b2 = cross(b3, np.broadcast_to(c1, b3.shape))
        b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
        b1 = cross(b2, b3)
        Rd = np.stack((b1, b2, b3), axis=2)
        collective = np.einsum("ni,ni->n", F, R[:, :, 2])
        collective = np.maximum(collective, 0.1 * self.mass * self.g)

        # Attitude error e_R = ½ vee(Rdᵀ R - Rᵀ Rd)
        E = np.einsum("nji,njk->nik", Rd, R) - np.einsum("nji,njk->nik", R, Rd)
        eR = 0.5 * np.stack((E[:, 2, 1], E[:, 0, 2], E[:, 1, 0]), axis=1)
        rate_sp = -self.kp_att * eR
        rate_sp[:, 2] = np.clip(rate_sp[:, 2], -2.0, 2.0)
        # Rate loop as a first-order response with time constant τ_rate:
        # desired angular acceleration = (ω_sp - ω) / τ_rate,  τ = I α + ω × I ω
        ang_acc = (rate_sp - omega) / self.rate_tau
        tau = ang_acc @ self.I.T + cross(omega, omega @ self.I.T)

        wrench = np.concatenate((collective[:, None], tau), axis=1)
        T = wrench @ self.A_inv.T
        frac = np.clip(T / self.t_max, 0.0, 1.0)
        h = env.nominal_hover
        return np.where(frac < h, frac / h - 1.0, (frac - h) / (1 - h))
