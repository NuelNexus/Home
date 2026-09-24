"""Autonomous waypoint missions with full-state recording (for logs, plots and videos)."""
from __future__ import annotations

import numpy as np

from .env import DroneEnv
from .math3d import quat_from_euler, quat_to_euler


def fly_mission(env: DroneEnv, act, waypoints, hold=2.0, timeout=8.0, wind=(0.0, 0.0, 0.0), throw=False,
                pushes=(), verbose=True):
    """Fly ``waypoints`` with controller ``act(obs) -> action`` in environment 0.

    ``pushes`` is a list of ``(t_seconds, (Fx, Fy, Fz))`` shoves lasting 0.15 s.
    Returns a dict of time-series arrays sampled at the control rate.
    """
    wps = np.asarray(waypoints, float)
    wind = np.asarray(wind, float)
    ph = env.phys
    env.reset()
    env.set_target(0, wps[0])
    tilt = (1.2, -0.6) if throw else (0.0, 0.0)
    rate = [4.0, -3.0, 2.0] if throw else [0.0, 0.0, 0.0]
    q0 = quat_from_euler(tilt[0], tilt[1], 0.0)
    ph.reset_state([0], wps[0] + [0, 0, 0.3], [0, 0, 0], q0, rate, ph.hover_rotor_speed()[0])
    f, w = ph.imu_kinematics()
    env.imu.reset([0], f[:1], w[:1])
    env.ahrs.reset([0], q0)
    env.pos_sensor.reset([0], ph.pos[:1], ph.vel[:1])
    env.wind_mean[:] = wind
    obs = env._observe()

    keys = ("t", "pos", "vel", "quat", "euler", "est_euler", "omega", "rotor", "action", "target", "imu_raw",
            "push", "wp_index")
    log = {k: [] for k in keys}
    events = []
    k, held, t_wp, t = 0, 0.0, 0.0, 0.0
    crashed = False
    while k < len(wps):
        u = act(obs)
        ph.wind[:] = wind
        ph.ext_force[:] = 0.0
        for tp, F in pushes:
            if tp <= t < tp + 0.15:
                ph.ext_force[0] = F
        obs, _, done, info = env.step(u)
        t += env.dt
        t_wp += env.dt
        log["t"].append(t)
        log["pos"].append(ph.pos[0].copy())
        log["vel"].append(ph.vel[0].copy())
        log["quat"].append(ph.quat[0].copy())
        log["euler"].append(quat_to_euler(ph.quat[0]))
        log["est_euler"].append(quat_to_euler(env.ahrs.q[0]))
        log["omega"].append(ph.omega[0].copy())
        log["rotor"].append(ph.rotor_speed[0].copy())
        log["action"].append(u[0].copy())
        log["target"].append(wps[k].copy())
        log["imu_raw"].append(env.imu.read_raw()[0].copy())
        log["push"].append(ph.ext_force[0].copy())
        log["wp_index"].append(k)
        if done[0] and info["crashed"][0]:
            crashed = True
            events.append((t, "CRASHED"))
            if verbose:
                print(f"CRASHED at t={t:.2f}s")
            break
        dist = np.linalg.norm(ph.pos[0] - wps[k])
        held = held + env.dt if dist < 0.25 else 0.0
        if held >= hold or t_wp > timeout:
            ok = held >= hold
            events.append((t, f"waypoint {k + 1} {'reached' if ok else 'timeout'}"))
            if verbose:
                print(f"  waypoint {k} {wps[k]} -> {'reached' if ok else 'TIMEOUT'} in {t_wp:.1f}s, error {dist:.3f} m")
            k, held, t_wp = k + 1, 0.0, 0.0
            if k < len(wps):
                env.set_target(0, wps[k])
    out = {key: np.asarray(v) for key, v in log.items()}
    out.update(waypoints=wps, events=events, crashed=crashed, dt=env.dt, wind=wind)
    return out
