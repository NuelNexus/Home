#!/usr/bin/env python3
"""Fly an autonomous waypoint mission and record everything (incl. raw MPU-6050 registers).

    python scripts/fly_mission.py --policy runs/f450/best.pt
    python scripts/fly_mission.py --controller pid --waypoints "0,0,1.5; 2,0,2; 2,2,2; 0,0,1"
    python scripts/fly_mission.py --policy runs/f450/best.pt --set payload=0.15 --wind 3 --throw
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from common import make_controller, make_model

from drone_sim import DroneEnv, EnvConfig
from drone_sim.math3d import quat_from_euler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy")
    p.add_argument("--controller", choices=["nn", "pid"])
    p.add_argument("--config")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--waypoints", default="0,0,1.5; 1.5,0,2; 1.5,1.5,2.5; -1,1.5,1.5; -1,-1,2; 0,0,1.5")
    p.add_argument("--hold", type=float, default=2.0, help="seconds to hold at each waypoint")
    p.add_argument("--timeout", type=float, default=8.0, help="max seconds per waypoint")
    p.add_argument("--wind", type=float, default=0.0, help="steady wind speed in m/s (along +x)")
    p.add_argument("--throw", action="store_true", help="start tumbling: 70° tilt, spinning")
    p.add_argument("--out", default="runs/mission")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    policy = None
    if a.policy:
        from drone_sim.rl import Policy
        policy = Policy.load(a.policy)
    kind = a.controller or ("nn" if policy else "pid")
    model = make_model(a, policy)
    wps = np.array([[float(v) for v in w.split(",")] for w in a.waypoints.split(";")])
    over = dict(randomize=False, waypoint_change_prob=0.0, episode_seconds=1e6, max_distance=50.0)
    cfg = policy.env_cfg(**over) if policy else EnvConfig.from_dict(over)
    env = DroneEnv(1, model, cfg, seed=a.seed)
    env.reset()
    env.set_target(0, wps[0])
    tilt = (1.2, -0.6) if a.throw else (0.0, 0.0)
    rate = [4.0, -3.0, 2.0] if a.throw else [0, 0, 0]
    q0 = quat_from_euler(tilt[0], tilt[1], 0.0)
    env.phys.reset_state([0], wps[0] + [0, 0, 0.3], [0, 0, 0], q0, rate, env.phys.hover_rotor_speed()[0])
    f, w = env.phys.imu_kinematics()
    env.imu.reset([0], f[:1], w[:1])
    env.ahrs.reset([0], q0)
    env.pos_sensor.reset([0], env.phys.pos[:1], env.phys.vel[:1])
    env.phys.wind[:] = [a.wind, 0, 0]
    env.wind_mean[:] = [a.wind, 0, 0]
    obs = env._observe()
    act, _ = make_controller(kind, env, policy)

    rows, k, held, t_wp, t = [], 0, 0.0, 0.0, 0.0
    reached = []
    print(f"flying {len(wps)} waypoints with the {kind.upper()} controller")
    while k < len(wps):
        u = act(obs)
        env.phys.wind[:] = [a.wind, 0, 0]
        obs, r, d, info = env.step(u)
        t += env.dt
        t_wp += env.dt
        s = env.true_state(0)
        dist = np.linalg.norm(s["pos"] - wps[k])
        rows.append([t, *s["pos"], *wps[k], *np.rad2deg(s["euler"]), *np.rad2deg(s["est_euler"]), *s["omega"],
                     *s["rotor_speed"], *u[0], *s["imu_raw"]])
        if d[0] and info["crashed"][0]:
            print(f"CRASHED at t={t:.2f}s")
            break
        held = held + env.dt if dist < 0.25 else 0.0
        if held >= a.hold or t_wp > a.timeout:
            reached.append((k, held >= a.hold, t_wp))
            print(f"  waypoint {k} {wps[k]} -> {'reached' if held >= a.hold else 'TIMEOUT'} in {t_wp:.1f}s, "
                  f"error {dist:.3f} m")
            k, held, t_wp = k + 1, 0.0, 0.0
            if k < len(wps):
                env.set_target(0, wps[k])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    header = (["t", "x", "y", "z", "tx", "ty", "tz", "roll", "pitch", "yaw", "est_roll", "est_pitch", "est_yaw",
               "p", "q", "r", "rotor0", "rotor1", "rotor2", "rotor3", "u0", "u1", "u2", "u3",
               "ACCEL_XOUT", "ACCEL_YOUT", "ACCEL_ZOUT", "TEMP_OUT", "GYRO_XOUT", "GYRO_YOUT", "GYRO_ZOUT"])
    with open(out / f"flight_{kind}.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([header] + rows)
    print(f"log written to {out / f'flight_{kind}.csv'}")
    try:
        plot(np.array(rows), out / f"flight_{kind}.png", kind)
        print(f"plot written to {out / f'flight_{kind}.png'}")
    except ImportError:
        print("(install matplotlib for plots)")


def plot(D, path, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = D[:, 0]
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(D[:, 1], D[:, 2], D[:, 3], lw=1.2, label="flight path")
    ax.scatter(D[:, 4], D[:, 5], D[:, 6], c="r", s=30, label="waypoints")
    ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]"), ax.set_zlabel("z [m]")
    ax.legend()
    ax.set_title(f"{kind.upper()} autonomous mission")
    ax = fig.add_subplot(2, 2, 2)
    for i, n in enumerate("xyz"):
        ax.plot(t, D[:, 1 + i], label=n)
        ax.plot(t, D[:, 4 + i], "--", color=ax.lines[-1].get_color(), lw=0.8)
    ax.set_title("position (dashed = target)"), ax.set_xlabel("s"), ax.legend()
    ax = fig.add_subplot(2, 2, 3)
    for i, n in enumerate(["roll", "pitch"]):
        ax.plot(t, D[:, 7 + i], label=f"true {n}")
        ax.plot(t, D[:, 10 + i], ":", label=f"MPU-6050 estimate {n}")
    ax.set_title("attitude [deg]"), ax.set_xlabel("s"), ax.legend()
    ax = fig.add_subplot(2, 2, 4)
    for i, n in enumerate(["ACCEL_XOUT", "ACCEL_YOUT", "ACCEL_ZOUT"]):
        ax.plot(t, D[:, 24 + i], lw=0.7, label=n)
    for i, n in enumerate(["GYRO_XOUT", "GYRO_YOUT", "GYRO_ZOUT"]):
        ax.plot(t, D[:, 28 + i], lw=0.7, label=n)
    ax.set_title("raw MPU-6050 registers [LSB]"), ax.set_xlabel("s"), ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
