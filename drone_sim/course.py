"""3D obstacle courses, collision-free path planning and path following.

The neural network is a low-level flight controller: it flies the drone to a
target using only the (simulated) MPU-6050 and a position fix. It has no idea
obstacles exist. Obstacle avoidance is handled one level up, the way real
autonomy stacks do it:

1. The course is a set of solid primitives (boxes and cylinders) with an
   exact signed-distance function (SDF).
2. A* on a 3D grid finds a path from start to goal through free space. Every
   obstacle is inflated by the drone's radius plus a safety margin, and cells
   close to obstacles cost extra so the path keeps its distance.
3. The path is shortcut (line of sight) and smoothed with a Catmull-Rom spline,
   and every sample of the smooth path is re-checked against the inflated SDF.
4. While flying, a "carrot" target slides along the path ahead of the drone,
   and the neural network chases it.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .env import DroneEnv
from .math3d import quat_from_euler, quat_to_euler


# --------------------------------------------------------------------- geometry
@dataclass
class Box:
    center: tuple
    size: tuple
    kind: str = "wall"

    def sdf(self, p):
        q = np.abs(p - np.asarray(self.center)) - np.asarray(self.size) / 2
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside

    def to_dict(self):
        return {"type": "box", "center": list(self.center), "size": list(self.size), "kind": self.kind}


@dataclass
class Cylinder:
    """Vertical cylinder standing on the ground (or starting at ``z0``)."""
    x: float
    y: float
    radius: float
    height: float
    z0: float = 0.0
    kind: str = "pillar"

    def sdf(self, p):
        d_xy = np.hypot(p[..., 0] - self.x, p[..., 1] - self.y) - self.radius
        zc = self.z0 + self.height / 2
        d_z = np.abs(p[..., 2] - zc) - self.height / 2
        outside = np.hypot(np.maximum(d_xy, 0.0), np.maximum(d_z, 0.0))
        inside = np.minimum(np.maximum(d_xy, d_z), 0.0)
        return outside + inside

    def to_dict(self):
        return {"type": "cylinder", "x": self.x, "y": self.y, "radius": self.radius, "height": self.height,
                "z0": self.z0, "kind": self.kind}


def wall_with_window(x, y_span, height, window_center, window_size, thickness=0.2):
    """A wall across the course (normal along x) with a rectangular window, as 4 boxes."""
    y0, y1 = y_span
    wy, wz = window_center
    ww, wh = window_size
    t = thickness
    parts = []
    # left and right of the window
    if wy - ww / 2 > y0:
        parts.append(Box((x, (y0 + wy - ww / 2) / 2, height / 2), (t, wy - ww / 2 - y0, height)))
    if wy + ww / 2 < y1:
        parts.append(Box((x, (wy + ww / 2 + y1) / 2, height / 2), (t, y1 - wy - ww / 2, height)))
    # below and above the window
    parts.append(Box((x, wy, (wz - wh / 2) / 2), (t, ww, wz - wh / 2)))
    parts.append(Box((x, wy, (wz + wh / 2 + height) / 2), (t, ww, height - wz - wh / 2)))
    return parts


@dataclass
class Course:
    obstacles: list
    bounds_lo: tuple
    bounds_hi: tuple
    start: tuple
    goal: tuple
    gates: list = field(default_factory=list)   # (center, size, label) for rendering the window frames
    labels: list = field(default_factory=list)  # (position, text)

    def sdf(self, p):
        p = np.asarray(p, float)
        d = np.full(p.shape[:-1], np.inf)
        for o in self.obstacles:
            d = np.minimum(d, o.sdf(p))
        return d

    def to_dict(self):
        return {"obstacles": [o.to_dict() for o in self.obstacles], "bounds_lo": list(self.bounds_lo),
                "bounds_hi": list(self.bounds_hi), "start": list(self.start), "goal": list(self.goal),
                "gates": [{"center": list(c), "size": list(s), "label": lab} for c, s, lab in self.gates],
                "labels": [{"pos": list(p), "text": t} for p, t in self.labels]}


def default_course() -> Course:
    """About 25 m course: pillar forest, window wall, low beam, second window, slalom, landing pad."""
    obs = []
    # 1. Pillar forest
    for x, y, r in [(3.0, -3.0, 0.35), (3.0, 0.0, 0.4), (3.0, 3.0, 0.35),
                    (5.0, -1.5, 0.4), (5.0, 1.5, 0.4),
                    (7.0, -3.0, 0.3), (7.0, 0.0, 0.35), (7.0, 3.0, 0.3)]:
        obs.append(Cylinder(x, y, r, 5.0))
    # 2. Wall with a high window, off to the left
    obs += wall_with_window(9.5, (-4, 4), 4.5, (1.6, 2.8), (2.2, 2.0))
    # 3. Low beam: a solid block you have to fly under
    obs.append(Box((12.5, 0.0, 3.35), (0.8, 8.0, 2.3), kind="beam"))
    obs.append(Cylinder(12.5, -3.7, 0.25, 2.2, kind="post"))
    obs.append(Cylinder(12.5, 3.7, 0.25, 2.2, kind="post"))
    # 4. Second wall with a low window on the right
    obs += wall_with_window(15.5, (-4, 4), 4.5, (-1.8, 1.5), (2.2, 1.8))
    # 5. Slalom: three barriers (nets between poles), each with a 2.4 m opening on alternating sides
    for i, x in enumerate([18.0, 19.8, 21.6]):
        yc = 1.6 if i % 2 == 0 else -1.6
        a, b = yc - 1.2, yc + 1.2
        obs.append(Cylinder(x, a, 0.12, 4.0, kind="pole"))
        obs.append(Cylinder(x, b, 0.12, 4.0, kind="pole"))
        if a > -4:
            obs.append(Box((x, (-4 + a) / 2, 2.0), (0.05, a + 4, 4.0), kind="net"))
        if b < 4:
            obs.append(Box((x, (b + 4) / 2, 2.0), (0.05, 4 - b, 4.0), kind="net"))
    gates = [((9.5, 1.6, 2.8), (2.2, 2.0), "GATE 1"), ((15.5, -1.8, 1.5), (2.2, 1.8), "GATE 2")]
    labels = [((5.0, 0.0, 5.4), "PILLAR FOREST"), ((12.5, 0.0, 4.9), "LOW BEAM"), ((19.8, 0.0, 4.5), "SLALOM")]
    return Course(obs, (-1.5, -4.0, 0.0), (25.0, 4.0, 4.5), start=(0.0, 0.0, 1.5), goal=(24.0, 0.0, 1.2),
                  gates=gates, labels=labels)


# --------------------------------------------------------------------- planning
def plan_path(course: Course, clearance=0.75, res=0.2, prefer=0.5, z_min=0.5):
    """A* on a 26-connected grid; cells within ``clearance`` of an obstacle are blocked.

    ``prefer`` adds cost for passing close (within clearance + prefer) so the
    path keeps extra distance where it can.
    """
    lo, hi = np.asarray(course.bounds_lo, float), np.asarray(course.bounds_hi, float)
    shape = tuple(np.floor((hi - lo) / res).astype(int) + 1)
    axes = [lo[i] + res * np.arange(shape[i]) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    d = course.sdf(grid)
    free = d > clearance
    free &= grid[..., 2] >= z_min
    free &= grid[..., 2] <= hi[2] - clearance * 0.5
    penalty = np.clip((clearance + prefer - d) / prefer, 0.0, 1.0) * 3.0

    def to_idx(p):
        return tuple(np.clip(np.round((np.asarray(p) - lo) / res).astype(int), 0, np.array(shape) - 1))

    s, g = to_idx(course.start), to_idx(course.goal)
    if not free[s] or not free[g]:
        raise ValueError("start or goal is inside an (inflated) obstacle")
    nbrs = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    steps = [np.sqrt(i * i + j * j + k * k) for i, j, k in nbrs]
    gscore = {s: 0.0}
    parent = {s: None}
    goal_np = np.array(g)
    openq = [(0.0, s)]
    closed = set()
    while openq:
        _, cur = heapq.heappop(openq)
        if cur in closed:
            continue
        if cur == g:
            break
        closed.add(cur)
        gc = gscore[cur]
        for (di, dj, dk), st in zip(nbrs, steps):
            n = (cur[0] + di, cur[1] + dj, cur[2] + dk)
            if not (0 <= n[0] < shape[0] and 0 <= n[1] < shape[1] and 0 <= n[2] < shape[2]):
                continue
            if not free[n] or n in closed:
                continue
            ng = gc + st * (1.0 + penalty[n])
            if ng < gscore.get(n, np.inf):
                gscore[n] = ng
                parent[n] = cur
                h = np.linalg.norm(np.array(n) - goal_np)
                heapq.heappush(openq, (ng + h, n))
    if g not in parent:
        raise RuntimeError("no collision-free path found")
    cells = []
    n = g
    while n is not None:
        cells.append(n)
        n = parent[n]
    pts = lo + res * np.array(cells[::-1], float)
    pts[0], pts[-1] = course.start, course.goal
    return pts


def _segment_clear(course, a, b, clearance, step=0.05):
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    s = a + np.linspace(0, 1, n)[:, None] * (b - a)
    return bool(np.all(course.sdf(s) > clearance))


def shortcut(course, pts, clearance):
    """Greedy line-of-sight shortcutting: keep only the points you need."""
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _segment_clear(course, pts[i], pts[j], clearance):
            j -= 1
        out.append(pts[j])
        i = j
    return np.array(out)


def catmull_rom(pts, spacing=0.05):
    """Centripetal Catmull-Rom spline through ``pts``, resampled evenly by arc length."""
    P = np.vstack((2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]))
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5 + 1e-9
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5 + 1e-9
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5 + 1e-9
        n = max(4, int(np.linalg.norm(p2 - p1) / 0.02))
        t = np.linspace(t1, t2, n, endpoint=False)[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    out.append(pts[-1:])
    dense = np.vstack(out)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    s_new = np.arange(0.0, s[-1], spacing)
    return np.stack([np.interp(s_new, s, dense[:, k]) for k in range(3)], axis=1)


def plan_smooth_path(course: Course, clearance=0.75, res=0.2):
    raw = plan_path(course, clearance, res)
    key = shortcut(course, raw, clearance)
    # Smooth; if the spline cuts a corner too close, add back intermediate A* points.
    for _ in range(6):
        smooth = catmull_rom(key)
        bad = course.sdf(smooth) <= clearance * 0.9
        if not bad.any():
            break
        # insert midpoints of the offending key segments from the raw A* path
        new = [key[0]]
        for a, b in zip(key[:-1], key[1:]):
            seg = (np.linalg.norm(smooth - a, axis=1) < np.linalg.norm(b - a) + 1e-6)
            ia = np.argmin(np.linalg.norm(raw - a, axis=1))
            ib = np.argmin(np.linalg.norm(raw - b, axis=1))
            if bad[seg].any() and ib - ia > 1:
                new.append(raw[(ia + ib) // 2])
            new.append(b)
        key = np.array(new)
    return raw, key, smooth


# ------------------------------------------------------------------ path flight
def fly_path(env: DroneEnv, act, path, course: Course | None = None, lookahead=0.9, speed=1.6, wind=(0, 0, 0),
             settle=1.5, max_time=60.0, drone_radius=0.36, throw=False):
    """Follow ``path`` (N x 3) with a carrot target and record everything.

    ``throw=True`` releases the drone tumbling (70° roll, spinning) at the start;
    the carrot waits until it has had time to level itself.
    """
    ph = env.phys
    wind = np.asarray(wind, float)
    env.reset()
    q0 = quat_from_euler(1.2, -0.6, 0.0) if throw else quat_from_euler(0.0, 0.0, 0.0)
    rate0 = [4.0, -3.0, 2.0] if throw else [0.0, 0.0, 0.0]
    ph.reset_state([0], path[0], [0, 0, 0], q0, rate0, ph.hover_rotor_speed()[0])
    t_start = 2.0 if throw else 0.5
    f, w = ph.imu_kinematics()
    env.imu.reset([0], f[:1], w[:1])
    env.ahrs.reset([0], q0)
    env.pos_sensor.reset([0], ph.pos[:1], ph.vel[:1])
    env.wind_mean[:] = wind
    env.set_target(0, path[0])
    obs = env._observe()
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s_path = np.concatenate(([0.0], np.cumsum(seg)))
    L = s_path[-1]
    s_carrot, t, t_goal = 0.0, 0.0, None
    keys = ("t", "pos", "vel", "quat", "euler", "est_euler", "omega", "rotor", "action", "target", "imu_raw",
            "clearance", "progress")
    log = {k: [] for k in keys}
    crashed = False
    while t < max_time:
        i_near = np.argmin(np.linalg.norm(path - ph.pos[0], axis=1))
        s_near = s_path[i_near]
        if t > t_start:
            s_carrot = min(s_carrot + speed * env.dt, s_near + lookahead, L)
        target = np.array([np.interp(s_carrot, s_path, path[:, k]) for k in range(3)])
        env.set_target(0, target)
        u = act(obs)
        ph.wind[:] = wind
        obs, _, done, info = env.step(u)
        t += env.dt
        clr = float(course.sdf(ph.pos[0]) - drone_radius) if course is not None else np.inf
        log["t"].append(t)
        log["pos"].append(ph.pos[0].copy())
        log["vel"].append(ph.vel[0].copy())
        log["quat"].append(ph.quat[0].copy())
        log["euler"].append(quat_to_euler(ph.quat[0]))
        log["est_euler"].append(quat_to_euler(env.ahrs.q[0]))
        log["omega"].append(ph.omega[0].copy())
        log["rotor"].append(ph.rotor_speed[0].copy())
        log["action"].append(u[0].copy())
        log["target"].append(target)
        log["imu_raw"].append(env.imu.read_raw()[0].copy())
        log["clearance"].append(clr)
        log["progress"].append(s_near / L)
        if clr < 0 or (done[0] and info["crashed"][0]):
            crashed = True
            break
        if s_carrot >= L and np.linalg.norm(ph.pos[0] - path[-1]) < 0.15:
            t_goal = t if t_goal is None else t_goal
            if t - t_goal > settle:
                break
    out = {k: np.asarray(v) for k, v in log.items()}
    out.update(path=path, crashed=crashed, dt=env.dt, wind=wind, path_length=L, t_goal=t_goal)
    return out
