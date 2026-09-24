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
from drone_sim.mission import fly_mission


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
    act, _ = make_controller(kind, env, policy)
    print(f"flying {len(wps)} waypoints with the {kind.upper()} controller")
    L = fly_mission(env, act, wps, a.hold, a.timeout, wind=(a.wind, 0, 0), throw=a.throw)
    rows = np.column_stack((L["t"], L["pos"], L["target"], np.rad2deg(L["euler"]), np.rad2deg(L["est_euler"]),
                            L["omega"], L["rotor"], L["action"], L["imu_raw"]))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    header = (["t", "x", "y", "z", "tx", "ty", "tz", "roll", "pitch", "yaw", "est_roll", "est_pitch", "est_yaw",
               "p", "q", "r", "rotor0", "rotor1", "rotor2", "rotor3", "u0", "u1", "u2", "u3",
               "ACCEL_XOUT", "ACCEL_YOUT", "ACCEL_ZOUT", "TEMP_OUT", "GYRO_XOUT", "GYRO_YOUT", "GYRO_ZOUT"])
    with open(out / f"flight_{kind}.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([header] + rows.tolist())
    print(f"log written to {out / f'flight_{kind}.csv'}")
    try:
        plot(rows, out / f"flight_{kind}.png", kind)
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
