"""Shared helpers for the command-line scripts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drone_sim import DroneEnv, DroneModel  # noqa: E402
from drone_sim.controllers import CascadedController  # noqa: E402


def make_model(args, policy=None) -> DroneModel:
    if getattr(args, "config", None):
        model = DroneModel.from_json(args.config)
    elif policy is not None:
        model = policy.drone_model()
    else:
        model = DroneModel.from_json()
    model.apply_overrides(getattr(args, "set", []))
    return model


def make_controller(kind, env, policy=None):
    if kind == "pid":
        ctl = CascadedController(env)
        return lambda obs: ctl(obs), ctl.reset
    return (lambda obs: policy(obs)), (lambda idx=None: None)


__all__ = ["DroneEnv", "DroneModel", "make_model", "make_controller"]
