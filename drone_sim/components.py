"""Component weight system.

A drone is described as a list of physical components (frame, arms, battery,
flight controller, MPU-6050, ESCs, motors, propellers, payload ...).  Each one
has a mass, a position in the body frame, a simple geometric shape and an
orientation.  From these the exact rigid-body mass properties are computed:

* total mass            ``M = Σ m_i``
* centre of mass        ``c = Σ m_i r_i / M``
* inertia about the CoM ``I = Σ [ R_i I_i R_iᵀ + m_i (|d_i|² E - d_i d_iᵀ) ]``
  with ``d_i = r_i - c`` (parallel-axis / Steiner theorem).

Moving the battery or adding a payload therefore shifts the CoM away from the
thrust centre, which produces a real torque imbalance the controller has to
fight -- exactly like on a real airframe.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "f450_mpu6050.json"


def _rot_rpy_deg(rpy_deg) -> np.ndarray:
    r, p, y = np.deg2rad(np.asarray(rpy_deg, float))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


@dataclass
class Component:
    """A single rigid part of the drone.

    ``shape`` is one of ``point``, ``box`` (size = [lx, ly, lz]),
    ``cylinder`` (size = [radius, height], axis along local z),
    ``sphere`` (size = [radius]) or ``rod`` (size = [length], along local x).
    """

    name: str
    mass: float
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    shape: str = "point"
    size: list = field(default_factory=list)
    orientation_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    role: str = ""
    group: str = ""

    def local_inertia(self) -> np.ndarray:
        """Inertia tensor about the component's own CoM, in its local axes."""
        m, s = self.mass, list(self.size)
        if self.shape == "point" or m == 0.0:
            return np.zeros((3, 3))
        if self.shape == "box":
            lx, ly, lz = s
            return m / 12.0 * np.diag([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])
        if self.shape == "cylinder":
            r, h = s
            ixx = m * (3 * r**2 + h**2) / 12.0
            return np.diag([ixx, ixx, 0.5 * m * r**2])
        if self.shape == "sphere":
            (r,) = s
            return np.eye(3) * 0.4 * m * r**2
        if self.shape == "rod":
            (length,) = s
            return np.diag([0.0, 1.0, 1.0]) * m * length**2 / 12.0
        raise ValueError(f"unknown shape '{self.shape}' for component '{self.name}'")

    def body_inertia(self) -> np.ndarray:
        """Local inertia rotated into body axes (still about the part's own CoM)."""
        R = _rot_rpy_deg(self.orientation_deg)
        return R @ self.local_inertia() @ R.T

    def to_dict(self) -> dict:
        d = {"name": self.name, "mass": self.mass, "position": list(map(float, self.position)), "shape": self.shape}
        if self.size:
            d["size"] = list(self.size)
        if any(self.orientation_deg):
            d["orientation_deg"] = list(self.orientation_deg)
        if self.role:
            d["role"] = self.role
        if self.group:
            d["group"] = self.group
        return d


@dataclass
class MassProperties:
    mass: float
    com: np.ndarray
    inertia: np.ndarray

    @property
    def inertia_inv(self) -> np.ndarray:
        return np.linalg.inv(self.inertia)


def compute_mass_properties(components: Iterable[Component]) -> MassProperties:
    comps = [c for c in components if c.mass > 0]
    if not comps:
        raise ValueError("drone has no mass")
    m = np.array([c.mass for c in comps])
    r = np.array([c.position for c in comps], float)
    M = m.sum()
    com = (m[:, None] * r).sum(0) / M
    I = np.zeros((3, 3))
    for mi, ri, c in zip(m, r, comps):
        d = ri - com
        I += c.body_inertia() + mi * (d @ d * np.eye(3) - np.outer(d, d))
    return MassProperties(float(M), com, I)


class DroneModel:
    """Full physical description of a multirotor built from components."""

    def __init__(self, config: dict):
        self.config = copy.deepcopy(config)
        self.name = config.get("name", "drone")
        self.gravity = float(config.get("gravity", 9.80665))
        self.rotor_cfg = config["rotors"]
        self.aero = config.get("aerodynamics", {})
        self.imu_cfg = config.get("imu", {})
        self.components: list[Component] = [Component(**c) for c in config["components"]]
        self._build_rotors()

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_json(cls, path: str | Path | None = None) -> "DroneModel":
        with open(path or DEFAULT_CONFIG) as f:
            return cls(json.load(f))

    def to_config(self) -> dict:
        cfg = copy.deepcopy(self.config)
        cfg["rotors"] = copy.deepcopy(self.rotor_cfg)
        cfg["components"] = [c.to_dict() for c in self.components if c.group != "rotor"]
        return cfg

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_config(), f, indent=2)

    def copy(self) -> "DroneModel":
        return DroneModel(self.to_config())

    # -------------------------------------------------------------- rotors
    def _build_rotors(self) -> None:
        rc = self.rotor_cfg
        n = int(rc.get("count", 4))
        L, h = float(rc["arm_length"]), float(rc.get("height", 0.0))
        layout = rc.get("layout", "x")
        if "positions" in rc:
            pos = np.asarray(rc["positions"], float)
        else:
            offset = np.pi / n if layout == "x" else 0.0
            # Motor 0 is front-left for an X quad, then counter-clockwise.
            ang = offset + 2 * np.pi * np.arange(n) / n
            pos = np.stack((L * np.cos(ang), L * np.sin(ang), np.full(n, h)), axis=1)
        self.rotor_positions = pos
        default_dirs = [1 if i % 2 == 0 else -1 for i in range(n)]
        self.spin_dirs = np.asarray(rc.get("spin_directions", default_dirs), float)
        motor_masses = rc.get("motor_masses", [rc.get("motor_mass", 0.0)] * n)
        prop_masses = rc.get("propeller_masses", [rc.get("propeller_mass", 0.0)] * n)
        r_m, h_m = rc.get("motor_radius", 0.014), rc.get("motor_height", 0.03)
        r_p = rc.get("propeller_radius", 0.127)
        self.components = [c for c in self.components if c.group != "rotor"]
        for i in range(n):
            self.components.append(
                Component(f"motor_{i}", float(motor_masses[i]), list(pos[i]), "cylinder", [r_m, h_m], group="rotor")
            )
            # A propeller is well approximated by a thin rod spinning about z;
            # we use its time-averaged inertia (a thin disc of the same mass).
            self.components.append(
                Component(
                    f"propeller_{i}", float(prop_masses[i]), list(pos[i] + [0, 0, h_m / 2]),
                    "cylinder", [r_p / np.sqrt(2), 0.005], group="rotor",
                )
            )

    @property
    def num_rotors(self) -> int:
        return len(self.rotor_positions)

    @property
    def kf(self) -> float:
        return float(self.rotor_cfg["thrust_coefficient"])

    @property
    def km(self) -> float:
        return float(self.rotor_cfg["torque_coefficient"])

    @property
    def max_rotor_speed(self) -> float:
        return float(self.rotor_cfg["max_speed_rad_s"])

    @property
    def max_thrust_per_rotor(self) -> float:
        return self.kf * self.max_rotor_speed**2

    # --------------------------------------------------- weight editing API
    def component(self, name: str) -> Component:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(f"no component named '{name}'. Available: {[c.name for c in self.components]}")

    def set_mass(self, name: str, mass_kg: float) -> None:
        """Set a component's mass in kg. ``motors`` / ``propellers`` set all rotors."""
        if mass_kg < 0:
            raise ValueError("mass must be >= 0")
        n = self.num_rotors
        if name in ("motors", "propellers") or name.startswith(("motor_", "propeller_")):
            kind = "motor" if name.startswith("motor") else "propeller"
            key = f"{kind}_masses"
            masses = list(self.rotor_cfg.get(key, [self.rotor_cfg.get(f"{kind}_mass", 0.0)] * n))
            if name.endswith("s"):
                masses = [mass_kg] * n
            else:
                masses[int(name.split("_")[1])] = mass_kg
            self.rotor_cfg[key] = masses
            self._build_rotors()
        else:
            self.component(name).mass = float(mass_kg)

    def scale_mass(self, name: str, factor: float) -> None:
        if name in ("motors", "propellers"):
            for c in self.components:
                if c.name.startswith(name[:-1] + "_"):
                    self.set_mass(c.name, c.mass * factor)
        else:
            self.set_mass(name, self.component(name).mass * factor)

    def move(self, name: str, position) -> None:
        self.component(name).position = list(map(float, position))

    def add_component(self, name: str, mass: float, position=(0, 0, 0), shape="point", size=(), **kw) -> None:
        if name in {c.name for c in self.components}:
            raise ValueError(f"component '{name}' already exists")
        self.components.append(Component(name, float(mass), list(position), shape, list(size), **kw))

    def remove_component(self, name: str) -> None:
        c = self.component(name)
        if c.group == "rotor":
            raise ValueError("rotor parts cannot be removed, set their mass instead")
        self.components.remove(c)

    def apply_overrides(self, overrides: Iterable[str]) -> None:
        """Apply CLI style overrides such as ``battery=0.25`` or ``payload@0.02,0,-0.05``."""
        for ov in overrides or []:
            if "@" in ov:
                name, pos = ov.split("@", 1)
                self.move(name.strip(), [float(x) for x in pos.split(",")])
            elif "=" in ov:
                name, val = ov.split("=", 1)
                self.set_mass(name.strip(), float(val))
            else:
                raise ValueError(f"cannot parse override '{ov}' (use name=kg or name@x,y,z)")

    # ------------------------------------------------------ derived values
    def mass_properties(self) -> MassProperties:
        return compute_mass_properties(self.components)

    @property
    def imu_position(self) -> np.ndarray:
        name = self.imu_cfg.get("component")
        if name:
            return np.asarray(self.component(name).position, float)
        return np.asarray(self.imu_cfg.get("position", [0, 0, 0]), float)

    def hover_thrust_fraction(self) -> float:
        mp = self.mass_properties()
        return mp.mass * self.gravity / (self.num_rotors * self.max_thrust_per_rotor)

    def thrust_to_weight(self) -> float:
        return 1.0 / self.hover_thrust_fraction()

    def validate(self) -> list[str]:
        """Return human-readable warnings about the physical configuration."""
        warns = []
        mp = self.mass_properties()
        tw = self.thrust_to_weight()
        if tw < 1.0:
            warns.append(f"thrust-to-weight {tw:.2f} < 1: the drone cannot take off")
        elif tw < 1.6:
            warns.append(f"thrust-to-weight {tw:.2f} is low (< 1.6): little control authority left")
        xy_off = np.linalg.norm(mp.com[:2])
        arm = np.min(np.linalg.norm(self.rotor_positions[:, :2], axis=1))
        if xy_off > 0.25 * arm:
            warns.append(f"CoM is {xy_off*1000:.0f} mm off-centre: large static torque imbalance")
        return warns

    def summary(self) -> str:
        mp = self.mass_properties()
        lines = [f"Drone: {self.name}", "-" * 68, f"{'component':<22}{'mass [g]':>10}{'x [mm]':>11}{'y [mm]':>11}{'z [mm]':>11}"]
        for c in self.components:
            x, y, z = (1000 * np.asarray(c.position)).tolist()
            lines.append(f"{c.name:<22}{c.mass*1000:>10.1f}{x:>11.1f}{y:>11.1f}{z:>11.1f}")
        I = mp.inertia
        lines += [
            "-" * 68,
            f"total mass            : {mp.mass*1000:.1f} g",
            "centre of mass [mm]   : " + ", ".join(f"{v*1000:.2f}" for v in mp.com),
            f"inertia diag [kg m^2] : {I[0,0]:.3e}, {I[1,1]:.3e}, {I[2,2]:.3e}",
            f"products of inertia   : Ixy={I[0,1]:.2e} Ixz={I[0,2]:.2e} Iyz={I[1,2]:.2e}",
            f"max thrust            : {self.num_rotors * self.max_thrust_per_rotor:.2f} N",
            f"thrust-to-weight      : {self.thrust_to_weight():.2f}",
            f"hover thrust fraction : {self.hover_thrust_fraction()*100:.1f} %  "
            f"(throttle ~{np.sqrt(self.hover_thrust_fraction())*100:.1f} % of max rotor speed)",
        ]
        for w in self.validate():
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)
