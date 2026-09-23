#!/usr/bin/env python3
"""Evaluate a trained policy (and/or the PID baseline) over many random episodes.

    python scripts/evaluate.py --policy runs/f450/best.pt
    python scripts/evaluate.py --policy runs/f450/best.pt --set payload=0.2      # heavier drone
    python scripts/evaluate.py --controller pid --no-randomize
"""
import argparse

import numpy as np
from common import make_controller, make_model

from drone_sim import DroneEnv


def evaluate(env, act, reset_ctl, episodes):
    obs = env.reset()
    reset_ctl()
    res = {"crashed": [], "final_distance": [], "return": []}
    while len(res["return"]) < episodes:
        obs, r, d, info = env.step(act(obs))
        if d.any():
            idx = info["done_idx"]
            reset_ctl(idx)
            res["crashed"] += list(info["crashed"][idx])
            res["final_distance"] += list(info["final_distance"])
            res["return"] += list(info["episode_return"])
    return {k: np.asarray(v[:episodes], float) for k, v in res.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", help="checkpoint .pt")
    p.add_argument("--controller", choices=["nn", "pid", "both"], default=None)
    p.add_argument("--config")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--envs", type=int, default=100)
    p.add_argument("--no-randomize", action="store_true", help="no wind, pushes or mass randomisation")
    p.add_argument("--tilt", type=float, help="max initial tilt in degrees")
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args()

    policy = None
    if a.policy:
        from drone_sim.rl import Policy
        policy = Policy.load(a.policy)
    kinds = {"both": ["nn", "pid"], None: ["nn", "pid"] if policy else ["pid"]}.get(a.controller, [a.controller])
    model = make_model(a, policy)
    print(model.summary(), "\n")
    over = {"waypoint_change_prob": 0.0}
    if a.no_randomize:
        over["randomize"] = False
    if a.tilt is not None:
        over["init_tilt_deg"] = a.tilt
    for kind in kinds:
        cfg = policy.env_cfg(**over) if policy else __import__("drone_sim").EnvConfig.from_dict(over)
        env = DroneEnv(a.envs, model, cfg, seed=a.seed)
        act, reset_ctl = make_controller(kind, env, policy)
        r = evaluate(env, act, reset_ctl, a.episodes)
        ok = ~r["crashed"].astype(bool)
        print(f"[{kind.upper():3s}] episodes {a.episodes} | crash rate {100*(1-ok.mean()):5.1f}% | "
              f"final distance (survivors) median {np.median(r['final_distance'][ok]) if ok.any() else np.nan:.3f} m, "
              f"p90 {np.percentile(r['final_distance'][ok], 90) if ok.any() else np.nan:.3f} m | "
              f"mean return {r['return'].mean():.1f}")


if __name__ == "__main__":
    main()
