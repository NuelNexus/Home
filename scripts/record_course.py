#!/usr/bin/env python3
"""Fly the neural-network autopilot through a 3D obstacle course and render it as a video.

Pipeline: A* path planner (obstacle avoidance) -> carrot target -> neural network
flies the drone from simulated MPU-6050 data -> three.js scene rendered frame by frame
in headless Chromium (solid meshes, lighting, soft shadows) -> MP4 via ffmpeg.

    python scripts/record_course.py                      # default course, docs/obstacle_course.mp4
    python scripts/record_course.py --set payload=0.2    # heavier drone
    python scripts/record_course.py --speed 2.0 --lookahead 1.2 --wind 1.5

Needs:  pip install imageio-ffmpeg   and   (cd render3d && npm install)
Chromium: set CHROMIUM_PATH, or it uses Playwright's bundled browser.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from common import make_model

from drone_sim import DroneEnv
from drone_sim.course import default_course, fly_path, plan_smooth_path

ROOT = Path(__file__).resolve().parents[1]


def section_name(x):
    for limit, name in [(1.5, "TAKE-OFF"), (8.3, "PILLAR FOREST"), (10.6, "GATE 1 · HIGH WINDOW"),
                        (14.0, "LOW BEAM · FLY UNDER"), (16.6, "GATE 2 · LOW WINDOW"), (22.6, "SLALOM")]:
        if x < limit:
            return name
    return "LANDING PAD"


def drone_geometry(model):
    mp = model.mass_properties()
    comps = []
    for c in model.components:
        if c.group == "rotor":
            continue
        comps.append({"name": c.name, "position": list(np.asarray(c.position) - mp.com), "shape": c.shape,
                      "size": list(c.size), "orientation_deg": list(c.orientation_deg), "mass": c.mass})
    rc = model.rotor_cfg
    return {"components": comps, "rotors": (model.rotor_positions - mp.com).tolist(),
            "spin": model.spin_dirs.tolist(), "prop_radius": float(rc.get("propeller_radius", 0.127)),
            "motor_radius": float(rc.get("motor_radius", 0.014)), "motor_height": float(rc.get("motor_height", 0.03)),
            "mass_g": mp.mass * 1000, "name": model.name}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default=str(ROOT / "models/f450_policy.pt"))
    p.add_argument("--config")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--speed", type=float, default=1.6, help="carrot speed along the path [m/s]")
    p.add_argument("--lookahead", type=float, default=0.9, help="max carrot lead [m]")
    p.add_argument("--wind", type=float, default=1.0, help="crosswind along +y [m/s]")
    p.add_argument("--clearance", type=float, default=0.75, help="planner clearance to obstacle surfaces [m]")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", default=str(ROOT / "docs/obstacle_course.mp4"))
    p.add_argument("--keep-json", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    from drone_sim.rl import Policy
    policy = Policy.load(a.policy)
    model = make_model(a, policy)
    course = default_course()
    print("planning collision-free path ...")
    raw, key, path = plan_smooth_path(course, clearance=a.clearance)
    print(f"  A* {len(raw)} cells -> {len(key)} key points -> smooth path {np.linalg.norm(np.diff(path, axis=0), axis=1).sum():.1f} m, "
          f"min distance to obstacles {course.sdf(path).min():.2f} m")
    env = DroneEnv(1, model, policy.env_cfg(randomize=False, waypoint_change_prob=0.0, episode_seconds=1e6,
                                            max_distance=50.0), seed=a.seed)
    print("flying (neural network, simulated MPU-6050) ...")
    L = fly_path(env, policy, path, course, lookahead=a.lookahead, speed=a.speed, wind=(0, a.wind, 0))
    dev = np.array([np.min(np.linalg.norm(path - q, axis=1)) for q in L["pos"]])
    status = "CRASHED" if L["crashed"] else "completed"
    print(f"  {status} in {L['t'][-1]:.1f} s | min clearance (prop tips to obstacle) {L['clearance'].min():.2f} m | "
          f"path deviation mean {dev.mean()*100:.0f} cm, max {dev.max()*100:.0f} cm | "
          f"top speed {np.linalg.norm(L['vel'], axis=1).max():.2f} m/s")

    # ---- resample the 100 Hz log at the video frame rate
    t = L["t"]
    n_frames = int(t[-1] * a.fps)
    idx = np.minimum((np.arange(n_frames) / a.fps / L["dt"]).astype(int), len(t) - 1)
    max_speed = model.max_rotor_speed
    frac = (L["rotor"] / max_speed) ** 2
    spin = np.cumsum(L["rotor"] * model.spin_dirs * L["dt"] * 0.035, axis=0)
    frames = []
    for i in idx:
        frames.append({
            "t": round(float(t[i]), 3), "p": np.round(L["pos"][i], 4).tolist(),
            "q": np.round(L["quat"][i], 5).tolist(), "v": np.round(L["vel"][i], 3).tolist(),
            "m": np.round(frac[i], 3).tolist(), "s": np.round(spin[i] % (2 * np.pi), 3).tolist(),
            "tg": np.round(L["target"][i], 3).tolist(), "c": round(float(L["clearance"][i]), 3),
            "e": np.round(np.rad2deg(L["euler"][i]), 1).tolist(),
            "ee": np.round(np.rad2deg(L["est_euler"][i]), 1).tolist(),
            "imu": L["imu_raw"][i].astype(int).tolist(), "pr": round(float(L["progress"][i]), 4),
            "sec": section_name(L["pos"][i][0]),
        })
    data = {
        "fps": a.fps, "width": a.width, "height": a.height, "course": course.to_dict(),
        "path": np.round(path[::4], 3).tolist(), "drone": drone_geometry(model), "frames": frames,
        "summary": {"status": status, "time": round(float(t[-1]), 1), "wind": a.wind,
                    "min_clearance": round(float(L["clearance"].min()), 2),
                    "dev_mean_cm": round(float(dev.mean() * 100)), "path_len": round(float(L["path_length"]), 1)},
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jpath = out.with_suffix(".json")
    jpath.write_text(json.dumps(data))
    import imageio_ffmpeg
    cmd = ["node", str(ROOT / "render3d/render.mjs"), str(jpath), str(out), imageio_ffmpeg.get_ffmpeg_exe()]
    print(f"rendering {n_frames} frames with three.js ...")
    r = subprocess.run(cmd, env=dict(os.environ))  # PREVIEW_FRAMES=... renders PNGs into --out instead
    if not a.keep_json:
        jpath.unlink(missing_ok=True)
    if r.returncode:
        sys.exit(r.returncode)
    print(f"video written to {out}")


if __name__ == "__main__":
    main()
