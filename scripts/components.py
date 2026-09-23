#!/usr/bin/env python3
"""Inspect and edit the drone's component weights.

    python scripts/components.py                                   # show the default drone
    python scripts/components.py --set battery=0.25 --set payload=0.1
    python scripts/components.py --set payload@0.03,0,-0.06        # move a component (m)
    python scripts/components.py --add camera=0.045@0.08,0,-0.01   # add a new part
    python scripts/components.py --remove receiver --save my_drone.json
    python scripts/components.py --config my_drone.json --set motors=0.07
"""
import argparse

from common import make_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--set", action="append", default=[], metavar="NAME=KG | NAME@X,Y,Z")
    p.add_argument("--add", action="append", default=[], metavar="NAME=KG@X,Y,Z")
    p.add_argument("--remove", action="append", default=[])
    p.add_argument("--save", help="write the edited configuration to this JSON file")
    a = p.parse_args()
    model = make_model(argparse.Namespace(config=a.config, set=[]))
    for spec in a.add:
        name, rest = spec.split("=", 1)
        mass, _, pos = rest.partition("@")
        model.add_component(name, float(mass), [float(x) for x in pos.split(",")] if pos else (0, 0, 0))
    for name in a.remove:
        model.remove_component(name)
    model.apply_overrides(a.set)
    print(model.summary())
    if a.save:
        model.save(a.save)
        print(f"\nsaved to {a.save}")


if __name__ == "__main__":
    main()
