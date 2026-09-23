"""Vectorised 6-DOF multirotor rigid-body dynamics.

State per drone (13 + n_rotors values)::

    p  (3)  position of the CoM in the world frame                [m]
    v  (3)  CoM velocity in the world frame                       [m/s]
    q  (4)  attitude quaternion, body -> world                    [-]
    w  (3)  angular velocity in the body frame                    [rad/s]
    Ω  (n)  rotor angular speeds                                   [rad/s]

Equations of motion (Newton-Euler)::

    ṗ = v
    m v̇ = R Σ T_i e_z + F_drag + F_wind + F_ground + m g
    q̇ = ½ q ⊗ [0, w]
    I ẇ = τ - w × (I w + h_rotors) - J_r Σ s_i Ω̇_i e_z
    Ω̇ = (Ω_cmd - Ω) / τ_motor                 (first-order ESC/motor lag)

    T_i = k_f Ω_i²                              rotor thrust
    τ   = Σ (r_i - c) × T_i e_z  -  Σ s_i k_m Ω_i² e_z  - D_w w
    h_rotors = J_r Σ s_i Ω_i e_z                rotor angular momentum

``s_i = +1`` means the rotor spins counter-clockwise seen from above, so its
aerodynamic reaction torque on the body is clockwise (``-z``).  The system is
integrated with classic 4th-order Runge-Kutta.
"""
from __future__ import annotations

import numpy as np

from .components import DroneModel
from .math3d import cross, matTvec, matvec, quat_mul, quat_normalize, quat_to_rotmat


class MultirotorPhysics:
    def __init__(self, num_envs: int, model: DroneModel, dt: float = 1e-3):
        self.n = num_envs
        self.dt = dt
        self.g = model.gravity
        self.nr = model.num_rotors
        self.spin = model.spin_dirs.astype(float)
        self.state = np.zeros((num_envs, 13 + self.nr))
        self.state[:, 6] = 1.0
        n, nr = num_envs, self.nr
        # Per-environment physical parameters (can be domain-randomised).
        self.mass = np.ones(n)
        self.inertia = np.tile(np.eye(3), (n, 1, 1))
        self.inertia_inv = np.tile(np.eye(3), (n, 1, 1))
        self.com = np.zeros((n, 3))
        self.rotor_rel = np.zeros((n, nr, 3))
        self.imu_rel = np.zeros((n, 3))
        self.kf = np.zeros((n, nr))
        self.km = np.zeros((n, nr))
        self.tau_motor = np.ones(n)
        self.rotor_inertia = np.zeros(n)
        self.max_speed = np.ones(n)
        self.lin_drag = np.zeros((n, 3))
        self.quad_drag = np.zeros((n, 3))
        self.rot_damp = np.zeros((n, 3))
        # Inputs held constant over a control period.
        self.rotor_cmd = np.zeros((n, nr))
        self.wind = np.zeros((n, 3))
        self.ext_force = np.zeros((n, 3))
        self.ext_torque = np.zeros((n, 3))
        # Quantities needed by the IMU model (evaluated at the current state).
        self.specific_force_body = np.zeros((n, 3))
        self.ang_accel_body = np.zeros((n, 3))
        self.ground_contact = np.zeros(n, bool)
        for i in range(n):
            self.set_params(i, model)

    # ------------------------------------------------------------ params
    def set_params(self, i: int, model: DroneModel, kf_scale=None, km_scale=None, tau_scale: float = 1.0,
                   drag_scale: float = 1.0) -> None:
        mp = model.mass_properties()
        self.mass[i] = mp.mass
        self.inertia[i] = mp.inertia
        self.inertia_inv[i] = np.linalg.inv(mp.inertia)
        self.com[i] = mp.com
        self.rotor_rel[i] = model.rotor_positions - mp.com
        self.imu_rel[i] = model.imu_position - mp.com
        self.kf[i] = model.kf * (1.0 if kf_scale is None else np.asarray(kf_scale))
        self.km[i] = model.km * (1.0 if km_scale is None else np.asarray(km_scale))
        rc = model.rotor_cfg
        self.tau_motor[i] = float(rc.get("time_constant", 0.03)) * tau_scale
        self.rotor_inertia[i] = float(rc.get("rotor_inertia", 0.0))
        self.max_speed[i] = model.max_rotor_speed
        aero = model.aero
        self.lin_drag[i] = np.asarray(aero.get("linear_drag", [0, 0, 0]), float) * drag_scale
        self.quad_drag[i] = np.asarray(aero.get("quadratic_drag", [0, 0, 0]), float) * drag_scale
        self.rot_damp[i] = np.asarray(aero.get("rotational_damping", [0, 0, 0]), float)

    # ------------------------------------------------------------ views
    @property
    def pos(self):
        return self.state[:, 0:3]

    @property
    def vel(self):
        return self.state[:, 3:6]

    @property
    def quat(self):
        return self.state[:, 6:10]

    @property
    def omega(self):
        return self.state[:, 10:13]

    @property
    def rotor_speed(self):
        return self.state[:, 13:]

    # ---------------------------------------------------------- dynamics
    def derivative(self, x: np.ndarray):
        """Return ``(ẋ, specific_force_body, ang_accel_body, contact)``."""
        v, q, w = x[:, 3:6], x[:, 6:10], x[:, 10:13]
        Om = np.maximum(x[:, 13:], 0.0)
        R = quat_to_rotmat(q)
        m = self.mass[:, None]

        Om_dot = (self.rotor_cmd - x[:, 13:]) / self.tau_motor[:, None]
        thrust = self.kf * Om**2                                   # (N, nr)
        F_body = np.zeros_like(v)
        F_body[:, 2] = thrust.sum(1)

        # Aerodynamic drag, computed from the velocity relative to the air mass.
        v_air_b = matTvec(R, v - self.wind)
        F_body -= self.lin_drag * v_air_b + self.quad_drag * np.abs(v_air_b) * v_air_b
        F_world = matvec(R, F_body) + self.ext_force

        # Torques about the CoM.
        rx, ry = self.rotor_rel[..., 0], self.rotor_rel[..., 1]
        tau = np.stack(
            (
                (ry * thrust).sum(1),
                -(rx * thrust).sum(1),
                -(self.spin * self.km * Om**2).sum(1),
            ),
            axis=1,
        )
        tau -= self.rot_damp * w
        tau += self.ext_torque
        Jr = self.rotor_inertia[:, None]
        h_rot = np.zeros_like(w)
        h_rot[:, 2] = (Jr * self.spin * Om).sum(1)
        tau[:, 2] -= (Jr * self.spin * Om_dot).sum(1)                # motor spin-up reaction

        # Ground contact: stiff spring-damper normal force + Coulomb-like friction.
        z, vz = x[:, 2], v[:, 2]
        contact = z <= 0.0
        if contact.any():
            k = self.mass * self.g / 0.002                             # 2 mm static compression
            c = 2.0 * np.sqrt(k * self.mass)                           # critical damping
            Fn = np.where(contact, np.maximum(0.0, -k * z - c * vz), 0.0)
            F_world[:, 2] += Fn
            F_world[:, :2] -= np.where(contact, 4.0 * self.mass, 0.0)[:, None] * v[:, :2]
            tau -= np.where(contact, 0.5, 0.0)[:, None] * w

        a = F_world / m
        a[:, 2] -= self.g
        Iw = matvec(self.inertia, w)
        w_dot = matvec(self.inertia_inv, tau - cross(w, Iw + h_rot))
        wq = np.concatenate((np.zeros((len(w), 1)), w), axis=1)
        q_dot = 0.5 * quat_mul(q, wq)

        xdot = np.concatenate((v, a, q_dot, w_dot, Om_dot), axis=1)
        specific_force_b = matTvec(R, F_world / m)
        return xdot, specific_force_b, w_dot, contact

    def step(self) -> None:
        """Advance every environment by one RK4 step of ``dt``."""
        dt, x = self.dt, self.state
        k1, *_ = self.derivative(x)
        k2, *_ = self.derivative(x + 0.5 * dt * k1)
        k3, *_ = self.derivative(x + 0.5 * dt * k2)
        k4, *_ = self.derivative(x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[:, 6:10] = quat_normalize(x[:, 6:10])
        x[:, 13:] = np.clip(x[:, 13:], 0.0, self.max_speed[:, None])
        self.state = x
        _, self.specific_force_body, self.ang_accel_body, self.ground_contact = self.derivative(x)

    def imu_kinematics(self):
        """Specific force and angular rate at the IMU location (body frame).

        The accelerometer is not at the CoM, so it also sees the tangential
        and centripetal accelerations ``α × r + ω × (ω × r)``.
        """
        w, r = self.omega, self.imu_rel
        f = self.specific_force_body + cross(self.ang_accel_body, r) + cross(w, cross(w, r))
        return f, w.copy()

    # ---------------------------------------------------------- helpers
    def rotation(self) -> np.ndarray:
        return quat_to_rotmat(self.quat)

    def hover_rotor_speed(self) -> np.ndarray:
        """Rotor speed that makes total thrust equal weight (per env, per rotor)."""
        per = self.mass * self.g / self.kf.sum(1)
        return np.sqrt(per)[:, None] * np.ones((1, self.nr))

    def reset_state(self, idx, pos, vel, quat, omega, rotor_speed) -> None:
        self.state[idx, 0:3] = pos
        self.state[idx, 3:6] = vel
        self.state[idx, 6:10] = quat
        self.state[idx, 10:13] = omega
        self.state[idx, 13:] = rotor_speed
        self.rotor_cmd[idx] = rotor_speed
        xdot, f, a, c = self.derivative(self.state)
        self.specific_force_body[idx] = f[idx]
        self.ang_accel_body[idx] = a[idx]
        self.ground_contact[idx] = c[idx]
