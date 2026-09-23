#!/usr/bin/env python3
"""Train a neural-network flight controller with PPO.

Examples
--------
    python scripts/train.py --steps 30e6
    python scripts/train.py --set battery=0.25 --set payload=0.15 --out runs/heavy
    python scripts/train.py --config my_drone.json --envs 512
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from drone_sim import DroneEnv, DroneModel, EnvConfig
from drone_sim.rl import PPO, PPOConfig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="drone JSON config (default: F450 + MPU-6050)")
    p.add_argument("--set", action="append", default=[], metavar="NAME=KG | NAME@X,Y,Z",
                   help="override a component's mass (kg) or position (m); repeatable")
    p.add_argument("--env-config", help="JSON file with EnvConfig overrides")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--steps", type=float, default=30e6)
    p.add_argument("--out", default="runs/default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", help="checkpoint to continue from")
    p.add_argument("--obs-mode", choices=["sensors", "state"], default="sensors")
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    model = DroneModel.from_json(a.config)
    model.apply_overrides(a.set)
    print(model.summary(), "\n")
    for w in model.validate():
        if "cannot take off" in w:
            sys.exit(f"refusing to train: {w}")
    env_over = json.load(open(a.env_config)) if a.env_config else {}
    env_over["observation_mode"] = a.obs_mode
    env = DroneEnv(a.envs, model, EnvConfig.from_dict(env_over), seed=a.seed)
    ppo = PPO(env, PPOConfig(), log_fn=lambda s: print(s, flush=True))
    if a.resume:
        ppo.load(a.resume)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save(Path(a.out) / "drone.json")
    history = ppo.train(int(a.steps), a.out)
    with open(Path(a.out) / "history.json", "w") as f:
        json.dump(history, f, default=float)
    print(f"done. checkpoints in {a.out}/best.pt and {a.out}/last.pt")


if __name__ == "__main__":
    main()
