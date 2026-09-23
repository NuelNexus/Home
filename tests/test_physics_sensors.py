import numpy as np
import pytest

from drone_sim import DroneEnv, DroneModel, EnvConfig
from drone_sim.components import Component, compute_mass_properties
from drone_sim.dynamics import MultirotorPhysics
from drone_sim.estimation import MahonyFilter
from drone_sim.math3d import quat_from_euler, quat_to_euler, quat_to_rotmat
from drone_sim.sensors import MPU6050, MPU6050Config
from drone_sim.sensors.mpu6050 import G0


@pytest.fixture
def model():
    return DroneModel.from_json()


# ------------------------------------------------------------------ components
def test_parallel_axis_two_point_masses():
    mp = compute_mass_properties([Component("a", 1.0, [1, 0, 0]), Component("b", 1.0, [-1, 0, 0])])
    assert mp.mass == 2.0
    np.testing.assert_allclose(mp.com, 0, atol=1e-12)
    np.testing.assert_allclose(np.diag(mp.inertia), [0, 2, 2])


def test_box_inertia_and_rotation():
    c = Component("box", 1.2, [0, 0, 0], "box", [0.3, 0.1, 0.05])
    I = c.local_inertia()
    np.testing.assert_allclose(I[0, 0], 1.2 / 12 * (0.1**2 + 0.05**2))
    c.orientation_deg = [0, 0, 90]  # rotate 90° about z -> Ixx and Iyy swap
    Ib = c.body_inertia()
    np.testing.assert_allclose([Ib[0, 0], Ib[1, 1]], [I[1, 1], I[0, 0]], atol=1e-12)


def test_weight_editing_moves_com(model):
    m0 = model.mass_properties()
    model.set_mass("payload", 0.3)
    model.move("payload", [0.05, 0.0, -0.06])
    m1 = model.mass_properties()
    assert m1.mass == pytest.approx(m0.mass + 0.3)
    expected_x = (m0.mass * m0.com[0] + 0.3 * 0.05) / m1.mass
    assert m1.com[0] == pytest.approx(expected_x)
    model.set_mass("motors", 0.07)
    assert all(model.component(f"motor_{i}").mass == 0.07 for i in range(4))
    model.apply_overrides(["battery=0.3", "battery@0.01,0,-0.03"])
    assert model.component("battery").mass == 0.3 and model.component("battery").position[0] == 0.01


def test_config_roundtrip(model, tmp_path):
    model.set_mass("battery", 0.21)
    model.save(tmp_path / "d.json")
    m2 = DroneModel.from_json(tmp_path / "d.json")
    np.testing.assert_allclose(m2.mass_properties().inertia, model.mass_properties().inertia)


def test_too_heavy_warns(model):
    model.set_mass("payload", 5.0)
    assert any("cannot take off" in w for w in model.validate())


# --------------------------------------------------------------------- physics
def _phys(model, n=1):
    ph = MultirotorPhysics(n, model, dt=1e-3)
    ph.lin_drag[:] = ph.quad_drag[:] = ph.rot_damp[:] = 0.0
    return ph


def test_free_fall_matches_analytic(model):
    ph = _phys(model)
    ph.reset_state([0], [0, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], np.zeros(4))
    for _ in range(1000):
        ph.step()
    assert ph.pos[0, 2] == pytest.approx(10 - 0.5 * model.gravity * 1.0, abs=1e-6)
    # Accelerometer reads zero specific force in free fall.
    np.testing.assert_allclose(ph.specific_force_body[0], 0, atol=1e-9)


def test_hover_equilibrium(model):
    model.move("battery", [0, 0, 0])  # keep CoM centred so torques cancel
    ph = _phys(model)
    ph.com[:] = 0
    ph.rotor_rel[0] = model.rotor_positions
    ph.reset_state([0], [0, 0, 2], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], ph.hover_rotor_speed()[0])
    for _ in range(2000):
        ph.step()
    np.testing.assert_allclose(ph.pos[0], [0, 0, 2], atol=1e-6)
    np.testing.assert_allclose(ph.specific_force_body[0], [0, 0, model.gravity], atol=1e-6)


def test_roll_torque_gives_expected_angular_acceleration(model):
    ph = _phys(model)
    ph.rotor_inertia[:] = 0
    Om = ph.hover_rotor_speed()[0].copy()
    Om[[0, 1]] *= 1.05   # left rotors (+y) faster -> positive roll torque
    Om[[2, 3]] *= 0.95
    ph.reset_state([0], [0, 0, 5], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], Om)
    xdot, _, wdot, _ = ph.derivative(ph.state)
    T = ph.kf[0] * Om**2
    tau = np.array([(ph.rotor_rel[0, :, 1] * T).sum(), -(ph.rotor_rel[0, :, 0] * T).sum(),
                    -(ph.spin * ph.km[0] * Om**2).sum()])
    np.testing.assert_allclose(wdot[0], np.linalg.solve(ph.inertia[0], tau), rtol=1e-9)
    assert wdot[0, 0] > 0


def test_ground_contact_supports_drone(model):
    ph = _phys(model)
    ph.reset_state([0], [0, 0, 0.0], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], np.zeros(4))
    for _ in range(3000):
        ph.step()
    assert abs(ph.pos[0, 2]) < 0.005
    assert ph.specific_force_body[0, 2] == pytest.approx(model.gravity, rel=1e-3)


def test_offset_imu_sees_centripetal_acceleration(model):
    ph = _phys(model)
    ph.imu_rel[0] = [0.1, 0.0, 0.0]
    ph.inertia[0] = np.diag(np.diag(ph.inertia[0]))   # principal axes -> no torque-free precession
    ph.inertia_inv[0] = np.linalg.inv(ph.inertia[0])
    ph.reset_state([0], [0, 0, 10], [0, 0, 0], [1, 0, 0, 0], [0, 0, 5.0], np.zeros(4))
    f, w = ph.imu_kinematics()
    # Free fall, spinning about z at 5 rad/s: a = ω×(ω×r) = -ω² r
    np.testing.assert_allclose(f[0], [-25 * 0.1, 0, 0], atol=1e-9)


# ---------------------------------------------------------------------- MPU6050
def _run_imu(imu, accel, gyro, seconds):
    n = int(seconds * imu.fs)
    out = []
    for _ in range(n):
        if imu.update(accel, gyro)[0]:
            out.append(imu.read_raw()[0].copy())
    return np.array(out)


def test_mpu6050_level_at_rest_register_values():
    cfg = MPU6050Config(accel_afs_sel=0, gyro_fs_sel=0, calibrated=True, calibrated_residual=0.0)
    imu = MPU6050(1, cfg, rng=np.random.default_rng(0))
    a, g = np.array([[0, 0, G0]]), np.zeros((1, 3))
    imu.reset([0], a, g)
    raw = _run_imu(imu, a, g, 2.0)
    assert raw[:, 2].mean() == pytest.approx(16384, abs=15)       # +1 g on Z at ±2 g range
    assert abs(raw[:, 0].mean()) < 15
    # Residual gyro offset is only temperature drift + g-sensitivity: < 0.25 °/s (33 LSB).
    assert np.all(np.abs(raw[:, 4:7].mean(0)) < 33)
    # TEMP_OUT decodes with the datasheet formula.
    assert imu.read_temperature()[0] == pytest.approx(imu.temperature[0], abs=1 / 340)
    assert imu.read_register(0, 0x75) == 0x68
    b = imu.burst_read(0)
    assert len(b) == 14 and int.from_bytes(b[4:6], "big", signed=True) == raw[-1, 2]


def test_mpu6050_noise_matches_datasheet():
    # Datasheet: gyro total RMS noise 0.05 °/s at 100 Hz bandwidth (0.005 °/s/√Hz).
    cfg = MPU6050Config(gyro_fs_sel=0, dlpf_cfg=2, smplrt_div=0, calibrated=True, calibrated_residual=0.0,
                        gyro_bias_walk_dps_per_sqrt_s=0.0, accel_bias_walk_mg_per_sqrt_s=0.0)
    imu = MPU6050(1, cfg, rng=np.random.default_rng(1))
    raw = _run_imu(imu, np.array([[0, 0, G0]]), np.zeros((1, 3)), 5.0)
    rms_dps = raw[:, 4:7].std(0) / 131.0
    assert np.all((rms_dps > 0.035) & (rms_dps < 0.075)), rms_dps


def test_mpu6050_saturates_and_quantises():
    imu = MPU6050(1, MPU6050Config(gyro_fs_sel=0, calibrated=True, calibrated_residual=0.0),
                  rng=np.random.default_rng(2))
    g = np.array([[np.deg2rad(400.0), 0, 0]])     # beyond ±250 °/s
    a = np.array([[0, 0, G0]])
    imu.reset([0], a, g)
    raw = _run_imu(imu, a, g, 0.2)
    assert raw[-1, 4] == 32767
    assert raw.dtype == np.int16


def test_mpu6050_dlpf_delays_step():
    cfg = MPU6050Config(dlpf_cfg=6, smplrt_div=0, calibrated=True, calibrated_residual=0.0)  # 5 Hz
    imu = MPU6050(1, cfg, rng=np.random.default_rng(3))
    a = np.array([[0, 0, G0]])
    imu.reset([0], a, np.zeros((1, 3)))
    step = np.array([[1.0, 0, 0]])  # 1 rad/s step
    raw = _run_imu(imu, a, step, 0.02)
    assert imu.read_gyro()[0, 0] < 0.3   # 5 Hz filter has barely responded after 20 ms


# --------------------------------------------------------------------- Mahony
def test_mahony_converges_to_true_attitude():
    q_true = quat_from_euler(0.3, -0.2, 0.0)[None]
    R = quat_to_rotmat(q_true)[0]
    accel = (R.T @ np.array([0, 0, G0]))[None]
    f = MahonyFilter(1, kp=2.0, ki=0.0)
    for _ in range(3000):
        f.update(np.zeros((1, 3)), accel, 0.002)
    np.testing.assert_allclose(quat_to_euler(f.q[0])[:2], [0.3, -0.2], atol=1e-3)


# ------------------------------------------------------------------------ env
def test_env_shapes_and_autoreset():
    env = DroneEnv(8, config=EnvConfig(episode_seconds=0.2), seed=0)
    obs = env.reset()
    assert obs.shape == (8, 25) and obs.dtype == np.float32
    for _ in range(25):
        obs, r, d, info = env.step(np.zeros((8, 4)))
        assert np.isfinite(obs).all() and np.isfinite(r).all()
    assert env.steps.max() < 20


def test_pid_baseline_flies():
    from drone_sim.controllers import CascadedController

    env = DroneEnv(16, config=EnvConfig(randomize=False, init_tilt_deg=20, waypoint_change_prob=0), seed=4)
    env.reset()
    ctl = CascadedController(env)
    for _ in range(600):
        _, _, d, _ = env.step(ctl())
        assert not d.any()
    dist = np.linalg.norm(env.phys.pos - env.target, axis=1)
    assert np.median(dist) < 0.15
