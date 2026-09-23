"""Datasheet-accurate InvenSense MPU-6050 6-axis IMU simulator.

References: *MPU-6000/MPU-6050 Product Specification* rev 3.4 (PS-MPU-6000A-00)
and *Register Map and Descriptions* rev 4.2 (RM-MPU-6000A-00).

Signal chain modelled for each axis (vectorised over N sensors)::

    true value (body frame, at the sensor's physical location)
      -> mounting rotation (body -> sensor axes)
      -> scale-factor error + cross-axis sensitivity   (M = (I + S) (I + C))
      -> + turn-on offset + temperature drift + bias random walk
      -> + gyro linear-acceleration (g) sensitivity
      -> + white noise   (datasheet noise spectral density)
      -> digital low-pass filter  (DLPF_CFG, 2nd-order Butterworth, 1 kHz)
      -> scale to LSB (FS_SEL / AFS_SEL), round, saturate to int16
      -> sample-rate divider  (SMPLRT_DIV, zero-order hold of the registers)

The accelerometer measures *specific force* ``f = a - g`` (so it reads +1 g
on z when level and at rest) and the gyroscope measures body angular rate.
The raw int16 registers can be read exactly as on the real chip, including a
14-byte big-endian burst read starting at ``ACCEL_XOUT_H`` (0x3B).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G0 = 9.80665  # the datasheet's "g"
DEG = np.pi / 180.0

# Full-scale range tables (register 0x1B FS_SEL / 0x1C AFS_SEL).
GYRO_FS_DPS = (250.0, 500.0, 1000.0, 2000.0)
GYRO_LSB_PER_DPS = (131.0, 65.5, 32.8, 16.4)
ACCEL_FS_G = (2.0, 4.0, 8.0, 16.0)
ACCEL_LSB_PER_G = (16384.0, 8192.0, 4096.0, 2048.0)

# DLPF_CFG (register 0x1A): (accel BW Hz, accel delay ms, gyro BW Hz, gyro delay ms, gyro Fs kHz)
DLPF_TABLE = {
    0: (260.0, 0.0, 256.0, 0.98, 8.0),
    1: (184.0, 2.0, 188.0, 1.9, 1.0),
    2: (94.0, 3.0, 98.0, 2.8, 1.0),
    3: (44.0, 4.9, 42.0, 4.8, 1.0),
    4: (21.0, 8.5, 20.0, 8.3, 1.0),
    5: (10.0, 13.8, 10.0, 13.4, 1.0),
    6: (5.0, 19.0, 5.0, 18.6, 1.0),
}

# Register addresses.
REG_SMPLRT_DIV, REG_CONFIG, REG_GYRO_CONFIG, REG_ACCEL_CONFIG = 0x19, 0x1A, 0x1B, 0x1C
REG_ACCEL_XOUT_H, REG_TEMP_OUT_H, REG_GYRO_XOUT_H, REG_WHO_AM_I = 0x3B, 0x41, 0x43, 0x75


@dataclass
class MPU6050Config:
    gyro_fs_sel: int = 1           # ±500 °/s
    accel_afs_sel: int = 2         # ±8 g
    dlpf_cfg: int = 3              # 44 Hz accel / 42 Hz gyro
    smplrt_div: int = 1            # 1 kHz / (1 + 1) = 500 Hz output rate
    internal_rate_hz: float = 1000.0
    orientation_deg: tuple = (0.0, 0.0, 0.0)   # sensor mounting (roll, pitch, yaw) relative to body
    # --- Datasheet noise / error figures --------------------------------------------------
    gyro_noise_density_dps: float = 0.005      # °/s/√Hz           (PS §6.1)
    accel_noise_density_ug: float = 400.0      # µg/√Hz            (PS §6.2)
    gyro_scale_tol: float = 0.03               # ±3 %  sensitivity scale factor tolerance
    accel_scale_tol: float = 0.03              # ±3 %
    cross_axis: float = 0.02                   # ±2 %  cross-axis sensitivity
    gyro_zro_dps: float = 20.0                 # ±20 °/s initial zero-rate output tolerance
    accel_offset_mg: tuple = (50.0, 50.0, 80.0)  # ±50 mg X/Y, ±80 mg Z zero-g offset
    gyro_g_sensitivity_dps_per_g: float = 0.1  # linear acceleration sensitivity
    gyro_zro_tempco_dps: float = 0.16          # ±20 °/s over -40..85 °C  -> ~0.16 °/s/°C worst case
    accel_offset_tempco_mg: tuple = (0.5, 0.5, 0.86)  # ±35 mg (X/Y), ±60 mg (Z) over 0..70 °C
    # --- Not in the datasheet: typical Allan-variance values for MEMS parts of this class
    gyro_bias_walk_dps_per_sqrt_s: float = 0.002
    accel_bias_walk_mg_per_sqrt_s: float = 0.05
    # --- After a standard power-on calibration (average N samples at rest) the remaining
    # errors are a small fraction of the datasheet tolerances.
    calibrated: bool = True
    calibrated_residual: float = 0.02          # remaining fraction of offsets / scale errors
    # --- Thermal model
    ambient_c: float = 25.0
    self_heating_c: float = 6.0
    thermal_time_constant_s: float = 60.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "MPU6050Config":
        d = dict(d or {})
        fields = cls.__dataclass_fields__
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in fields})


def _butterworth2(fc: float, fs: float):
    """2nd-order Butterworth low-pass biquad (bilinear transform with pre-warping)."""
    K = np.tan(np.pi * fc / fs)
    Q = 1.0 / np.sqrt(2.0)
    norm = 1.0 / (1.0 + K / Q + K * K)
    b0 = K * K * norm
    b = np.array([b0, 2 * b0, b0])
    a = np.array([1.0, 2 * (K * K - 1) * norm, (1 - K / Q + K * K) * norm])
    return b, a


class _Biquad:
    """Vectorised transposed direct-form II biquad, state shape (N, C)."""

    def __init__(self, b, a, shape):
        self.b, self.a = b, a
        self.z1 = np.zeros(shape)
        self.z2 = np.zeros(shape)

    def reset(self, idx, value):
        # Initialise at steady state for a constant input ``value``.
        b, a = self.b, self.a
        self.z1[idx] = value * (1.0 - b[0])
        self.z2[idx] = value * (b[2] - a[2])

    def __call__(self, x):
        b, a = self.b, self.a
        y = b[0] * x + self.z1
        self.z1 = b[1] * x - a[1] * y + self.z2
        self.z2 = b[2] * x - a[2] * y
        return y


def _rot(rpy_deg):
    r, p, y = np.deg2rad(rpy_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class MPU6050:
    """N independent simulated MPU-6050 chips."""

    WHO_AM_I = 0x68

    def __init__(self, num: int, config: MPU6050Config | None = None, rng: np.random.Generator | None = None):
        self.n = num
        self.cfg = cfg = config or MPU6050Config()
        self.rng = rng or np.random.default_rng()
        if cfg.dlpf_cfg not in DLPF_TABLE:
            raise ValueError("dlpf_cfg must be 0..6")
        self.fs = cfg.internal_rate_hz
        self.dt = 1.0 / self.fs
        a_bw, _, g_bw, _, _ = DLPF_TABLE[cfg.dlpf_cfg]
        self.accel_bw, self.gyro_bw = a_bw, g_bw
        self.gyro_sens = GYRO_LSB_PER_DPS[cfg.gyro_fs_sel] / DEG          # LSB per rad/s
        self.accel_sens = ACCEL_LSB_PER_G[cfg.accel_afs_sel] / G0          # LSB per m/s²
        self.output_rate_hz = self.fs / (1 + cfg.smplrt_div)
        # Sensor axes relative to body axes: v_sensor = R_sb @ v_body
        self.R_sb = _rot(cfg.orientation_deg).T
        # Pre-filter white-noise standard deviation per internal sample: σ = ND · √(fs/2)
        self.gyro_sigma = cfg.gyro_noise_density_dps * DEG * np.sqrt(self.fs / 2)
        self.accel_sigma = cfg.accel_noise_density_ug * 1e-6 * G0 * np.sqrt(self.fs / 2)
        ga, aa = _butterworth2(g_bw, self.fs), _butterworth2(a_bw, self.fs)
        self._gyro_lpf = _Biquad(*ga, (num, 3))
        self._accel_lpf = _Biquad(*aa, (num, 3))
        self.raw = np.zeros((num, 7), dtype=np.int16)       # ax ay az temp gx gy gz
        self._tick = np.zeros(num, dtype=np.int64)
        self.temperature = np.full(num, cfg.ambient_c)
        # Per-chip manufacturing errors -- sampled in reset().
        self.gyro_M = np.tile(np.eye(3), (num, 1, 1))
        self.accel_M = np.tile(np.eye(3), (num, 1, 1))
        self.gyro_bias = np.zeros((num, 3))
        self.accel_bias = np.zeros((num, 3))
        self.gyro_tc = np.zeros((num, 3))
        self.accel_tc = np.zeros((num, 3))
        self.gyro_walk = np.zeros((num, 3))
        self.accel_walk = np.zeros((num, 3))
        self.reset(np.arange(num))

    # --------------------------------------------------------------- setup
    def reset(self, idx, accel_body=None, gyro_body=None) -> None:
        """Re-sample per-chip errors (a "new chip") and settle the filters."""
        idx = np.atleast_1d(np.asarray(idx))
        k = len(idx)
        if k == 0:
            return
        c, rng = self.cfg, self.rng
        frac = c.calibrated_residual if c.calibrated else 1.0

        def mis(tol):
            # Scale-factor error on the diagonal, cross-axis coupling off-diagonal,
            # both uniform within the datasheet tolerance.
            S = np.tile(np.eye(3), (k, 1, 1))
            S[:, [0, 1, 2], [0, 1, 2]] += rng.uniform(-tol, tol, (k, 3)) * frac
            off = rng.uniform(-c.cross_axis, c.cross_axis, (k, 3, 3)) * frac
            off[:, [0, 1, 2], [0, 1, 2]] = 0.0
            return S + off

        self.gyro_M[idx] = mis(c.gyro_scale_tol)
        self.accel_M[idx] = mis(c.accel_scale_tol)
        self.gyro_bias[idx] = rng.uniform(-1, 1, (k, 3)) * c.gyro_zro_dps * DEG * frac
        self.accel_bias[idx] = rng.uniform(-1, 1, (k, 3)) * np.asarray(c.accel_offset_mg) * 1e-3 * G0 * frac
        self.gyro_tc[idx] = rng.uniform(-1, 1, (k, 3)) * c.gyro_zro_tempco_dps * DEG * max(frac, 0.1)
        self.accel_tc[idx] = rng.uniform(-1, 1, (k, 3)) * np.asarray(c.accel_offset_tempco_mg) * 1e-3 * G0 * max(frac, 0.1)
        self.gyro_walk[idx] = 0.0
        self.accel_walk[idx] = 0.0
        self.temperature[idx] = c.ambient_c + c.self_heating_c * rng.uniform(0.5, 1.0, k)
        self._tick[idx] = 0
        a0 = np.zeros((k, 3)) if accel_body is None else accel_body
        g0 = np.zeros((k, 3)) if gyro_body is None else gyro_body
        a_s, g_s = self._corrupt(idx, a0, g0, noise=False)
        self._accel_lpf.reset(idx, a_s)
        self._gyro_lpf.reset(idx, g_s)
        self.raw[idx] = self._quantise(idx, a_s, g_s)

    # -------------------------------------------------------------- signal
    def _corrupt(self, idx, accel_body, gyro_body, noise=True):
        c = self.cfg
        a = accel_body @ self.R_sb.T
        g = gyro_body @ self.R_sb.T
        dT = (self.temperature[idx] - 25.0)[:, None]
        a_meas = np.einsum("kij,kj->ki", self.accel_M[idx], a) + self.accel_bias[idx] \
            + self.accel_walk[idx] + self.accel_tc[idx] * dT
        g_meas = np.einsum("kij,kj->ki", self.gyro_M[idx], g) + self.gyro_bias[idx] \
            + self.gyro_walk[idx] + self.gyro_tc[idx] * dT
        g_meas += c.gyro_g_sensitivity_dps_per_g * DEG / G0 * a
        if noise:
            a_meas += self.rng.standard_normal(a_meas.shape) * self.accel_sigma
            g_meas += self.rng.standard_normal(g_meas.shape) * self.gyro_sigma
        return a_meas, g_meas

    def _quantise(self, idx, a, g):
        out = np.empty((len(a), 7), dtype=np.int16)
        out[:, 0:3] = np.clip(np.round(a * self.accel_sens), -32768, 32767)
        out[:, 4:7] = np.clip(np.round(g * self.gyro_sens), -32768, 32767)
        # TEMP_OUT: T[°C] = TEMP_OUT / 340 + 36.53
        out[:, 3] = np.clip(np.round((self.temperature[idx] - 36.53) * 340.0), -32768, 32767)
        return out

    def update(self, accel_body: np.ndarray, gyro_body: np.ndarray) -> np.ndarray:
        """Advance one internal sample (1 kHz) with true specific force [m/s²]
        and angular rate [rad/s] at the sensor. Returns the mask of chips whose
        output registers were refreshed this tick."""
        c, idx = self.cfg, np.arange(self.n)
        dt = self.dt
        # Bias random walks and die temperature (first-order warm-up).
        self.gyro_walk += self.rng.standard_normal((self.n, 3)) * c.gyro_bias_walk_dps_per_sqrt_s * DEG * np.sqrt(dt)
        self.accel_walk += self.rng.standard_normal((self.n, 3)) * c.accel_bias_walk_mg_per_sqrt_s * 1e-3 * G0 * np.sqrt(dt)
        target_T = c.ambient_c + c.self_heating_c
        self.temperature += (target_T - self.temperature) * dt / c.thermal_time_constant_s
        a, g = self._corrupt(idx, accel_body, gyro_body)
        a_f = self._accel_lpf(a)
        g_f = self._gyro_lpf(g)
        self._tick += 1
        ready = (self._tick % (1 + c.smplrt_div)) == 0
        if ready.any():
            self.raw[ready] = self._quantise(idx[ready], a_f[ready], g_f[ready])
        return ready

    # ---------------------------------------------------------- driver API
    def read_raw(self) -> np.ndarray:
        """int16 array (N, 7): ACCEL_X/Y/Z, TEMP, GYRO_X/Y/Z exactly as in registers 0x3B..0x48."""
        return self.raw.copy()

    def read_accel(self) -> np.ndarray:
        """Accelerometer in m/s² (what a driver computes from the raw LSBs)."""
        return self.raw[:, 0:3].astype(float) / self.accel_sens

    def read_gyro(self) -> np.ndarray:
        """Gyroscope in rad/s."""
        return self.raw[:, 4:7].astype(float) / self.gyro_sens

    def read_temperature(self) -> np.ndarray:
        return self.raw[:, 3].astype(float) / 340.0 + 36.53

    def burst_read(self, i: int = 0) -> bytes:
        """14-byte I2C burst read from ACCEL_XOUT_H (0x3B), big-endian like the chip."""
        return self.raw[i].astype(">i2").tobytes()

    def read_register(self, i: int, addr: int) -> int:
        """Read one 8-bit register of chip ``i`` (data, config and WHO_AM_I registers)."""
        c = self.cfg
        if addr == REG_WHO_AM_I:
            return self.WHO_AM_I
        if addr == REG_SMPLRT_DIV:
            return c.smplrt_div & 0xFF
        if addr == REG_CONFIG:
            return c.dlpf_cfg & 0x07
        if addr == REG_GYRO_CONFIG:
            return (c.gyro_fs_sel & 0x3) << 3
        if addr == REG_ACCEL_CONFIG:
            return (c.accel_afs_sel & 0x3) << 3
        if 0x3B <= addr <= 0x48:
            return self.burst_read(i)[addr - 0x3B]
        raise KeyError(f"register 0x{addr:02X} not simulated")
