"""Vectorised reinforcement-learning environment for autonomous flight.

The agent controls the four motors directly (100 Hz) and only sees what a
real flight controller sees: MPU-6050 readings (through the Mahony attitude
filter) plus a noisy external position/velocity fix.  Rewards are computed
from the true simulator state.

Every episode starts from a random "thrown" state (tilted up to
``init_tilt_deg``, spinning, moving), so the policy learns to self-balance and
recover before flying to its target.  Component masses, motor constants, drag
and wind are randomised so the policy is robust to changes in the weight
configuration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .components import DroneModel
from .dynamics import MultirotorPhysics
from .estimation import MahonyFilter
from .math3d import quat_from_euler, quat_to_rotmat
from .sensors import MPU6050, MPU6050Config, PositionSensor


@dataclass
class EnvConfig:
    control_hz: float = 100.0
    physics_hz: float = 1000.0
    episode_seconds: float = 10.0
    observation_mode: str = "sensors"          # "sensors" (realistic) or "state" (privileged)
    # Targets / task
    target_low: tuple = (-2.0, -2.0, 1.0)
    target_high: tuple = (2.0, 2.0, 3.0)
    waypoint_change_prob: float = 0.003        # per control step -> new random target mid-flight
    obs_pos_clip: float = 2.0
    # Initial "throw" disturbance
    init_pos_range: float = 1.0
    init_vel_range: float = 1.0
    init_tilt_deg: float = 60.0
    init_rate_range: float = 3.0
    # Domain randomisation
    mass_randomization: float = 0.15           # ±15 % on every component's mass
    position_jitter: float = 0.01              # ±1 cm on component positions
    motor_randomization: float = 0.05          # ±5 % thrust/torque coefficient per rotor
    motor_tau_randomization: float = 0.2
    drag_randomization: float = 0.3
    wind_max: float = 2.0                      # m/s mean wind
    gust_std: float = 0.7                      # m/s, Ornstein-Uhlenbeck gusts
    push_prob: float = 0.004                   # per step chance of a random shove
    push_force: float = 6.0                    # N
    # Position sensor
    pos_sensor_rate_hz: float = 50.0
    pos_noise: float = 0.02
    vel_noise: float = 0.05
    # Termination
    max_distance: float = 5.0
    max_rate: float = 35.0
    min_height: float = 0.03
    crash_penalty: float = 10.0
    randomize: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "EnvConfig":
        d = dict(d or {})
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


OBS_DIM = 25
ACT_DIM = 4


class DroneEnv:
    def __init__(self, num_envs: int = 1, model: DroneModel | None = None, config: EnvConfig | None = None,
                 seed: int | None = None):
        self.n = num_envs
        self.model = model or DroneModel.from_json()
        self.cfg = cfg = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.substeps = int(round(cfg.physics_hz / cfg.control_hz))
        self.dt = 1.0 / cfg.control_hz
        self.max_steps = int(round(cfg.episode_seconds * cfg.control_hz))
        self.g = self.model.gravity
        self.nr = self.model.num_rotors
        assert self.nr == ACT_DIM, "the RL policy is built for quadrotors"

        self.phys = MultirotorPhysics(num_envs, self.model, dt=1.0 / cfg.physics_hz)
        imu_cfg = MPU6050Config.from_dict(self.model.imu_cfg)
        imu_cfg.internal_rate_hz = cfg.physics_hz
        self.imu = MPU6050(num_envs, imu_cfg, rng=self.rng)
        self.ahrs = MahonyFilter(num_envs, g=self.g)
        self.pos_sensor = PositionSensor(num_envs, cfg.pos_sensor_rate_hz, (cfg.pos_noise,) * 3, cfg.vel_noise,
                                         rng=self.rng)
        # Action mapping uses the *configured* (nominal) drone, like real firmware would.
        self.nominal_hover = self.model.hover_thrust_fraction()

        self.target = np.zeros((num_envs, 3))
        self.prev_action = np.zeros((num_envs, ACT_DIM))
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.wind_mean = np.zeros((num_envs, 3))
        self.gust = np.zeros((num_envs, 3))
        self.push_timer = np.zeros(num_envs)
        self.ep_return = np.zeros(num_envs)
        self.obs = np.zeros((num_envs, OBS_DIM), np.float32)

    # ----------------------------------------------------------- helpers
    def action_to_rotor_cmd(self, action: np.ndarray) -> np.ndarray:
        """Map a ∈ [-1, 1] to a thrust fraction with a = 0 at nominal hover.

        Piecewise linear: a = -1 -> 0 thrust, 0 -> hover, +1 -> max thrust.
        Rotor speed follows Ω = Ω_max √(thrust fraction) because T = k_f Ω².
        """
        a = np.clip(action, -1.0, 1.0)
        h = self.nominal_hover
        frac = np.where(a < 0, h * (1 + a), h + a * (1 - h))
        return np.sqrt(frac) * self.phys.max_speed[:, None]

    def _randomize_physics(self, i: int) -> None:
        cfg, rng = self.cfg, self.rng
        if not cfg.randomize:
            self.phys.set_params(i, self.model)
            return
        m = self.model.copy()
        for c in m.components:
            if c.group == "rotor":
                continue
            c.mass *= 1 + rng.uniform(-cfg.mass_randomization, cfg.mass_randomization)
            c.position = list(np.asarray(c.position) + rng.uniform(-cfg.position_jitter, cfg.position_jitter, 3))
        m.scale_mass("motors", 1 + rng.uniform(-0.05, 0.05))
        mr = cfg.motor_randomization
        self.phys.set_params(
            i, m,
            kf_scale=1 + rng.uniform(-mr, mr, self.nr),
            km_scale=1 + rng.uniform(-mr, mr, self.nr),
            tau_scale=1 + rng.uniform(-cfg.motor_tau_randomization, cfg.motor_tau_randomization),
            drag_scale=1 + rng.uniform(-cfg.drag_randomization, cfg.drag_randomization),
        )

    def _sample_target(self, k: int) -> np.ndarray:
        return self.rng.uniform(self.cfg.target_low, self.cfg.target_high, (k, 3))

    def reset_envs(self, idx) -> None:
        idx = np.atleast_1d(np.asarray(idx))
        if len(idx) == 0:
            return
        cfg, rng, k = self.cfg, self.rng, len(idx)
        for i in idx:
            self._randomize_physics(int(i))
        self.target[idx] = self._sample_target(k)
        pos = self.target[idx] + rng.uniform(-cfg.init_pos_range, cfg.init_pos_range, (k, 3))
        pos[:, 2] = np.maximum(pos[:, 2], 0.5)
        vel = rng.uniform(-cfg.init_vel_range, cfg.init_vel_range, (k, 3))
        tilt = np.deg2rad(cfg.init_tilt_deg)
        quat = quat_from_euler(rng.uniform(-tilt, tilt, k), rng.uniform(-tilt, tilt, k), rng.uniform(-np.pi, np.pi, k))
        omega = rng.uniform(-cfg.init_rate_range, cfg.init_rate_range, (k, 3))
        self.phys.reset_state(idx, pos, vel, quat, omega, self.phys.hover_rotor_speed()[idx])
        if cfg.randomize:
            direction = rng.standard_normal((k, 3)) * np.array([1, 1, 0.2])
            direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9
            self.wind_mean[idx] = direction * rng.uniform(0, cfg.wind_max, (k, 1))
        else:
            self.wind_mean[idx] = 0.0
        self.gust[idx] = 0.0
        f, w = self.phys.imu_kinematics()
        self.imu.reset(idx, f[idx], w[idx])
        # The attitude filter has been running before the throw: small initial error.
        err = quat_from_euler(*(rng.normal(0, np.deg2rad(2), (3, k))))
        from .math3d import quat_mul
        self.ahrs.reset(idx, quat_mul(quat, err))
        self.pos_sensor.reset(idx, pos, vel)
        self.prev_action[idx] = 0.0
        self.steps[idx] = 0
        self.ep_return[idx] = 0.0

    def reset(self) -> np.ndarray:
        self.reset_envs(np.arange(self.n))
        self.obs = self._observe()
        return self.obs.copy()

    def set_target(self, idx, target) -> None:
        self.target[idx] = target

    # ------------------------------------------------------- observation
    def _observe(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.observation_mode == "state":
            R = self.phys.rotation()
            gyro = self.phys.omega
            acc = self.phys.imu_kinematics()[0]
            pos, vel = self.phys.pos, self.phys.vel
        else:
            R = self.ahrs.rotation()
            gyro = self.imu.read_gyro()
            acc = self.imu.read_accel()
            pos, vel = self.pos_sensor.pos, self.pos_sensor.vel
        err = np.clip(self.target - pos, -cfg.obs_pos_clip, cfg.obs_pos_clip)
        obs = np.concatenate(
            (R.reshape(self.n, 9), gyro / np.pi, acc / self.g, err, vel / 2.0, self.prev_action), axis=1
        )
        return obs.astype(np.float32)

    # --------------------------------------------------------------- step
    def step(self, action: np.ndarray):
        cfg, rng, ph = self.cfg, self.rng, self.phys
        action = np.clip(np.asarray(action, float), -1, 1)
        ph.rotor_cmd = self.action_to_rotor_cmd(action)

        # Wind: mean + Ornstein-Uhlenbeck gusts (1 s correlation time).
        if cfg.randomize:
            theta = self.dt / 1.0
            self.gust += -theta * self.gust + cfg.gust_std * np.sqrt(2 * theta) * rng.standard_normal((self.n, 3))
            ph.wind = self.wind_mean + self.gust
            # Random shoves (someone bumping the drone) lasting 0.1 s.
            new_push = (rng.random(self.n) < cfg.push_prob) & (self.push_timer <= 0)
            if new_push.any():
                k = int(new_push.sum())
                ph.ext_force[new_push] = rng.standard_normal((k, 3)) * cfg.push_force
                self.push_timer[new_push] = 0.1
            self.push_timer -= self.dt
            ph.ext_force[self.push_timer <= 0] = 0.0
        else:
            # No random disturbances: keep any steady wind / external force set by the caller.
            ph.wind[:] = self.wind_mean

        dt_phys = ph.dt
        for _ in range(self.substeps):
            ph.step()
            f, w = ph.imu_kinematics()
            ready = self.imu.update(f, w)
            if ready.any():
                self.ahrs.update(self.imu.read_gyro(), self.imu.read_accel(),
                                 dt_phys * (1 + self.imu.cfg.smplrt_div), mask=ready)
        self.pos_sensor.update(self.dt, ph.pos, ph.vel)

        if cfg.waypoint_change_prob > 0:
            change = rng.random(self.n) < cfg.waypoint_change_prob
            if change.any():
                self.target[change] = self._sample_target(int(change.sum()))

        reward, crashed = self._reward(action)
        self.prev_action = action
        self.steps += 1
        truncated = (self.steps >= self.max_steps) & ~crashed
        done = crashed | truncated
        self.ep_return += reward

        info = {"crashed": crashed, "truncated": truncated}
        obs = self._observe()
        if done.any():
            idx = np.nonzero(done)[0]
            info["terminal_obs"] = obs[idx].copy()
            info["done_idx"] = idx
            info["episode_return"] = self.ep_return[idx].copy()
            info["episode_length"] = self.steps[idx].copy()
            info["final_distance"] = np.linalg.norm(ph.pos[idx] - self.target[idx], axis=1)
            self.reset_envs(idx)
            obs[idx] = self._observe()[idx]
        self.obs = obs
        return obs.copy(), reward.astype(np.float32), done, info

    def _reward(self, action):
        cfg, ph = self.cfg, self.phys
        e = np.linalg.norm(ph.pos - self.target, axis=1)
        speed = np.linalg.norm(ph.vel, axis=1)
        rate = np.linalg.norm(ph.omega, axis=1)
        up = quat_to_rotmat(ph.quat)[:, 2, 2]          # cos(tilt)
        r = (
            1.5
            - 0.5 * np.minimum(e, 3.0)
            + 0.5 * np.exp(-(e / 0.2) ** 2)
            - 0.05 * speed
            - 0.05 * rate
            - 0.3 * (1.0 - up)
            - 0.02 * np.mean(action**2, axis=1)
            - 0.1 * np.mean((action - self.prev_action) ** 2, axis=1)
        )
        crashed = (ph.pos[:, 2] < cfg.min_height) | (e > cfg.max_distance) | (rate > cfg.max_rate)
        r = np.where(crashed, -cfg.crash_penalty, r)
        return r, crashed

    # ---------------------------------------------------------- logging
    def true_state(self, i: int = 0) -> dict:
        from .math3d import quat_to_euler
        ph = self.phys
        return {
            "pos": ph.pos[i].copy(), "vel": ph.vel[i].copy(), "euler": quat_to_euler(ph.quat[i]),
            "omega": ph.omega[i].copy(), "rotor_speed": ph.rotor_speed[i].copy(), "target": self.target[i].copy(),
            "est_euler": quat_to_euler(self.ahrs.q[i]), "imu_raw": self.imu.read_raw()[i].copy(),
        }
