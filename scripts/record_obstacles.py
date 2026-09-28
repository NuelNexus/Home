#!/usr/bin/env python3
"""Record the neural-network drone flying the obstacle course, as a 3D matplotlib video.

Same layout as record_flight.py (scene view, true-scale chase camera, attitude plot,
MPU-6050 telemetry), with solid shaded obstacles and a solid, shaded drone body.

    python scripts/record_obstacles.py                          # docs/obstacle_flight.mp4
    python scripts/record_obstacles.py --no-throw --wind 1.5
    python scripts/record_obstacles.py --set payload=0.2 --out runs/heavy_course.mp4
"""
import argparse
from pathlib import Path

import numpy as np
from common import make_model
from record_flight import BG, DIM, FG, FRONT, PANEL, REAR, style3d

from drone_sim import DroneEnv
from drone_sim.course import Box, Cylinder, default_course, fly_path, plan_smooth_path
from drone_sim.math3d import quat_to_rotmat

LIGHT = np.array([-0.35, -0.55, 0.76]) / np.linalg.norm([-0.35, -0.55, 0.76])
COL = {"pillar": "#b9b4aa", "wall": "#cfc9bd", "beam": "#e8b923", "post": "#8a9099", "pole": "#e0413a",
       "net": "#ff8a1f", "frame": "#ff9d2e", "pad": "#3a3f48"}


def rgba(hex_color, alpha=1.0):
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [alpha])


def shade(poly, base, center, ambient=0.38):
    """Lambert-shade a planar polygon; the normal is flipped to point away from ``center``."""
    n = np.cross(poly[1] - poly[0], poly[2] - poly[0])
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return base
    n /= nn
    if np.dot(n, poly.mean(0) - center) < 0:
        n = -n
    k = ambient + (1 - ambient) * max(0.0, float(n @ LIGHT))
    c = base.copy()
    c[:3] = np.clip(c[:3] * k, 0, 1)
    return c


# ------------------------------------------------------------------ primitive meshes
def box_polys(lo, hi, tile=None):
    """Faces of an axis-aligned box, optionally split into tiles (helps depth sorting)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    polys = []
    for ax in range(3):
        for side in (lo[ax], hi[ax]):
            u, v = [a for a in range(3) if a != ax]
            nu = 1 if tile is None else max(1, int(np.ceil((hi[u] - lo[u]) / tile)))
            nv = 1 if tile is None else max(1, int(np.ceil((hi[v] - lo[v]) / tile)))
            us = np.linspace(lo[u], hi[u], nu + 1)
            vs = np.linspace(lo[v], hi[v], nv + 1)
            for a in range(nu):
                for b in range(nv):
                    q = np.zeros((4, 3))
                    q[:, ax] = side
                    q[:, u] = [us[a], us[a + 1], us[a + 1], us[a]]
                    q[:, v] = [vs[b], vs[b], vs[b + 1], vs[b + 1]]
                    polys.append(q)
    return polys


def cyl_polys(x, y, r, z0, z1, n=18, dz=0.9):
    ang = np.linspace(0, 2 * np.pi, n + 1)
    zs = np.linspace(z0, z1, max(2, int(np.ceil((z1 - z0) / dz)) + 1))
    polys = []
    for i in range(n):
        a0, a1 = ang[i], ang[i + 1]
        for j in range(len(zs) - 1):
            polys.append(np.array([[x + r * np.cos(a0), y + r * np.sin(a0), zs[j]],
                                   [x + r * np.cos(a1), y + r * np.sin(a1), zs[j]],
                                   [x + r * np.cos(a1), y + r * np.sin(a1), zs[j + 1]],
                                   [x + r * np.cos(a0), y + r * np.sin(a0), zs[j + 1]]]))
    polys.append(np.stack([x + r * np.cos(ang[:-1]), y + r * np.sin(ang[:-1]), np.full(n, z1)], 1))
    return polys


def course_mesh(course):
    """All obstacle polygons with shaded colours, plus per-polygon x-extent for view culling."""
    polys, cols = [], []

    def add(ps, color, center, alpha=1.0, pole_bands=False):
        base = rgba(color, alpha)
        for p in ps:
            c = base
            if pole_bands:
                c = rgba("#f2f2f2" if int(p[:, 2].mean() / 0.5) % 2 else color, alpha)
            polys.append(p)
            cols.append(shade(p, c, center))

    for o in course.obstacles:
        if isinstance(o, Cylinder):
            n = 10 if o.radius < 0.2 else 18
            ps = cyl_polys(o.x, o.y, o.radius, o.z0, o.z0 + o.height, n=n, dz=0.5 if o.kind == "pole" else 0.9)
            add(ps, COL.get(o.kind, "#aaaaaa"), np.array([o.x, o.y, o.z0 + o.height / 2]),
                pole_bands=o.kind == "pole")
            if o.kind == "pillar":  # hazard band
                band = cyl_polys(o.x, o.y, o.radius * 1.03, 1.9, 2.3, n=n, dz=0.4)[:-1]
                add(band, "#f2c31b", np.array([o.x, o.y, 2.1]))
        elif isinstance(o, Box):
            c, sz = np.asarray(o.center), np.asarray(o.size)
            alpha = {"net": 0.25, "wall": 0.55, "beam": 0.6}.get(o.kind, 1.0)
            add(box_polys(c - sz / 2, c + sz / 2, tile=0.75), COL.get(o.kind, COL["wall"]), c, alpha)
    for (gx, gy, gz), (w, h), _ in course.gates:   # glowing window frames
        t = 0.08
        for lo, hi in [((gx - 0.16, gy - w / 2 - t, gz + h / 2), (gx + 0.16, gy + w / 2 + t, gz + h / 2 + t)),
                       ((gx - 0.16, gy - w / 2 - t, gz - h / 2 - t), (gx + 0.16, gy + w / 2 + t, gz - h / 2)),
                       ((gx - 0.16, gy - w / 2 - t, gz - h / 2), (gx + 0.16, gy - w / 2, gz + h / 2)),
                       ((gx - 0.16, gy + w / 2, gz - h / 2), (gx + 0.16, gy + w / 2 + t, gz + h / 2))]:
            add(box_polys(lo, hi, tile=0.75), COL["frame"], (np.asarray(lo) + hi) / 2)
    for p in (course.start, course.goal):   # take-off / landing pads
        add(cyl_polys(p[0], p[1], 0.8, 0.0, 0.03, n=24, dz=1), COL["pad"], np.array([p[0], p[1], 0.0]))
    xmin = np.array([p[:, 0].min() for p in polys])
    xmax = np.array([p[:, 0].max() for p in polys])
    return polys, np.array(cols), xmin, xmax


# ------------------------------------------------------------------ solid drone
def drone_parts(model):
    """Body-frame solid parts: list of (polygons, colour, part centre)."""
    com = model.mass_properties().com
    rotors = model.rotor_positions - com
    parts = []

    def oriented_box(p0, p1, width, thick, color, z=0.0):
        d = (p1 - p0)[:2]
        d = np.r_[d / np.linalg.norm(d), 0]
        s = np.array([-d[1], d[0], 0])
        e = np.array([0, 0, 1.0])
        c = [p0 + a * s * width / 2 + b * e * thick / 2 + e * z for a in (-1, 1) for b in (-1, 1)]
        c += [p1 + a * s * width / 2 + b * e * thick / 2 + e * z for a in (-1, 1) for b in (-1, 1)]
        c = np.array(c)
        faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
        return [c[f] for f in faces], c.mean(0)

    def abox(center, size, color):
        center, size = np.asarray(center, float) - com, np.asarray(size, float)
        return box_polys(center - size / 2, center + size / 2), center

    for rp in rotors:
        polys, cen = oriented_box(np.array([0, 0, rp[2] - 0.012]), rp * [1, 1, 0] + [0, 0, rp[2] - 0.012],
                                  0.034, 0.024, None)
        parts.append((polys, FRONT if rp[0] > 0 else REAR, cen))
    comps = {c.name: c for c in model.components}
    body = comps.get("center_plates")
    if body is not None:
        polys, cen = abox(body.position, [0.15, 0.15, 0.032], None)
        parts.append((polys, "#353b45", cen))
    for name, color in [("battery", "#2457c5"), ("flight_controller", "#138a45"), ("mpu6050", "#2f7bff"),
                        ("receiver", "#1c1c1c")]:
        c = comps.get(name)
        if c is not None and c.mass > 0:
            size = np.maximum(np.asarray(c.size, float), [0.012, 0.012, 0.006])
            polys, cen = abox(c.position, size, None)
            parts.append((polys, color, cen))
    for rp in rotors:
        polys = cyl_polys(rp[0], rp[1], 0.024, rp[2] - 0.012, rp[2] + 0.028, n=12, dz=1)
        bottom = cyl_polys(rp[0], rp[1], 0.024, rp[2] - 0.012, rp[2] - 0.012, n=12, dz=1)[-1:]
        parts.append((polys + bottom, "#c9ced6", rp.copy()))
    return parts, rotors


def drone_polys(parts, rotors, pos, R, frac, spin, prop_r, scale=1.0, shadow=True):
    polys, cols = [], []
    W = lambda p: pos + scale * (p @ R.T)  # noqa: E731
    for ps, color, cen in parts:
        base = rgba(color)
        cw = W(cen)
        for p in ps:
            pw = W(p)
            polys.append(pw)
            cols.append(shade(pw, base, cw, ambient=0.42))
    ang = np.linspace(0, 2 * np.pi, 25)[:-1]
    for k, rp in enumerate(rotors):
        top = rp + [0, 0, 0.032]
        disc = np.stack([top[0] + prop_r * np.cos(ang), top[1] + prop_r * np.sin(ang), np.full(24, top[2])], 1)
        col = rgba(FRONT if rp[0] > 0 else "#9aa3ad", 0.16 + 0.35 * min(1.0, frac[k] * 2.2))
        polys.append(W(disc))
        cols.append(col)
        th = spin[k]
        d = np.array([np.cos(th), np.sin(th), 0.0])
        s = np.array([-d[1], d[0], 0.0])
        for sign in (1, -1):   # two thick blades
            tip, root = top + sign * d * prop_r, top + sign * d * 0.02
            blade = np.array([root - s * 0.012, tip - s * 0.008, tip + s * 0.008, root + s * 0.012])
            polys.append(W(blade))
            cols.append(rgba("#f0f0f0", 0.95))
    if shadow:   # soft ground shadow: project the solid parts straight down
        for ps, _, _ in parts[:5]:
            for p in ps:
                pw = W(p)
                pw[:, 2] = 0.004
                polys.append(pw)
                cols.append(np.array([0, 0, 0, 0.18]))
    return polys, cols


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--policy", default=str(root / "models/f450_policy.pt"))
    p.add_argument("--config")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--wind", type=float, default=1.0, help="crosswind along +y [m/s]")
    p.add_argument("--speed", type=float, default=1.6)
    p.add_argument("--lookahead", type=float, default=0.9)
    p.add_argument("--no-throw", action="store_true", help="start level instead of thrown at 70°")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", default=str(root / "docs/obstacle_flight.mp4"))
    p.add_argument("--preview", help="comma-separated times [s]: save PNG stills next to --out instead of a video")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    from drone_sim.rl import Policy
    policy = Policy.load(a.policy)
    model = make_model(a, policy)
    course = default_course()
    print("planning collision-free path ...")
    _, _, path = plan_smooth_path(course)
    env = DroneEnv(1, model, policy.env_cfg(randomize=False, waypoint_change_prob=0.0, episode_seconds=1e6,
                                            max_distance=50.0), seed=a.seed)
    print("flying (neural network, simulated MPU-6050) ...")
    L = fly_path(env, policy, path, course, lookahead=a.lookahead, speed=a.speed, wind=(0, a.wind, 0),
                 throw=not a.no_throw)
    dev = np.array([np.min(np.linalg.norm(path - q, axis=1)) for q in L["pos"]])
    print(f"  {'CRASHED' if L['crashed'] else 'completed'} in {L['t'][-1]:.1f} s | min clearance "
          f"{L['clearance'].min():.2f} m | mean path deviation {dev.mean()*100:.0f} cm")
    render(L, model, course, path, a)


def section_name(x):
    for limit, name in [(1.5, "TAKE-OFF"), (8.3, "PILLAR FOREST"), (10.6, "GATE 1: HIGH WINDOW"),
                        (14.0, "LOW BEAM: FLY UNDER"), (16.6, "GATE 2: LOW WINDOW"), (22.6, "SLALOM")]:
        if x < limit:
            return name
    return "LANDING PAD"


def render(L, model, course, path, a):
    import imageio_ffmpeg
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    t, pos, quat, dt = L["t"], L["pos"], L["quat"], L["dt"]
    frac = (L["rotor"] / model.max_rotor_speed) ** 2
    spin = np.cumsum(L["rotor"] * model.spin_dirs * dt * 0.02, axis=0)
    n_frames = int(t[-1] * a.fps)
    times = np.arange(n_frames) / a.fps
    if a.preview:
        times = np.array([float(x) for x in a.preview.split(",")])
        n_frames = len(times)
    idx = np.minimum((times / dt).astype(int), len(t) - 1)
    euler, est = np.rad2deg(L["euler"]), np.rad2deg(L["est_euler"])
    prop_r = float(model.rotor_cfg.get("propeller_radius", 0.127))

    cpolys, ccols, cxmin, cxmax = course_mesh(course)
    parts, rotors = drone_parts(model)
    lo_c, hi_c = np.asarray(course.bounds_lo), np.asarray(course.bounds_hi)
    VIEW_BACK, VIEW_FWD = 3.5, 6.0

    dpi = 100
    fig = plt.figure(figsize=(a.width / dpi, a.height / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([-0.02, 0.15, 0.66, 0.78], projection="3d")
    axc = fig.add_axes([0.62, 0.40, 0.38, 0.55], projection="3d")
    axa = fig.add_axes([0.665, 0.07, 0.315, 0.27])
    axm = fig.add_axes([0.43, 0.02, 0.20, 0.13])
    fig.text(0.012, 0.965, f"Neural-network autopilot: obstacle course  ·  {model.name}", color=FG, fontsize=13,
             weight="bold", va="center")
    mp = model.mass_properties()
    fig.text(0.012, 0.93, f"mass {mp.mass*1000:.0f} g   ·   thrust/weight {model.thrust_to_weight():.2f}   ·   "
             f"crosswind {a.wind:.1f} m/s   ·   sensors: simulated MPU-6050 + position fix   ·   A* path planner",
             color=DIM, fontsize=8.5, va="center")
    hud = fig.text(0.015, 0.02, "", color=FG, fontsize=8.3, family="monospace", va="bottom")
    banner = fig.text(0.31, 0.885, "", color="#ffcf4d", fontsize=12, weight="bold", ha="center")

    # Static minimap (top view)
    axm.set_facecolor(PANEL)
    for o in course.obstacles:
        if isinstance(o, Cylinder):
            axm.add_patch(plt.Circle((o.x, o.y), max(o.radius, 0.12), color="#8d96a3" if o.kind != "pole" else "#e0413a"))
        else:
            c, s = o.center, o.size
            axm.add_patch(plt.Rectangle((c[0] - s[0] / 2, c[1] - s[1] / 2), max(s[0], 0.15), s[1],
                                        color=COL["net"] if o.kind == "net" else "#8d96a3",
                                        alpha=0.5 if o.kind in ("net", "beam") else 1.0))
    axm.plot(path[:, 0], path[:, 1], color="#35d6ff", lw=0.8)
    trail_m, = axm.plot([], [], color="#ffc53d", lw=1.4)
    dot_m, = axm.plot([], [], "o", color="white", ms=4)
    axm.set_xlim(lo_c[0], hi_c[0]), axm.set_ylim(lo_c[1], hi_c[1]), axm.set_aspect("equal")
    axm.set_xticks([]), axm.set_yticks([])
    for s in axm.spines.values():
        s.set_color("#2a323d")
    axm.set_title("top view", color=DIM, fontsize=7, pad=2)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if not a.preview:
        writer = imageio_ffmpeg.write_frames(str(out), (a.width, a.height), fps=a.fps, quality=8, macro_block_size=8)
        writer.send(None)
    cx = pos[0, 0]
    for f, i in enumerate(idx):
        R = quat_to_rotmat(quat[i])
        cx = pos[i, 0] if a.preview else cx + (pos[i, 0] - cx) * (1 - np.exp(-(1 / a.fps) / 0.6))  # smoothed view centre
        x0, x1 = max(lo_c[0], cx - VIEW_BACK), cx + VIEW_FWD
        # ---------------------------------------------------------------- scene view (follows the drone)
        ax.cla()
        style3d(ax)
        ax.set_xlim(x0, x1), ax.set_ylim(lo_c[1], hi_c[1]), ax.set_zlim(0, hi_c[2])
        ax.set_box_aspect((x1 - x0, hi_c[1] - lo_c[1], hi_c[2]))
        ax.set_xlabel("x [m]", fontsize=8), ax.set_ylabel("y [m]", fontsize=8), ax.set_zlabel("z [m]", fontsize=8)
        ax.view_init(elev=32, azim=-58 + 8 * np.sin(t[i] / 5.0))
        ax.computed_zorder = False
        keep = (cxmax > x0) & (cxmin < x1)
        polys = []
        for k in np.nonzero(keep)[0]:
            q = cpolys[k].copy()
            q[:, 0] = np.clip(q[:, 0], x0, x1)
            polys.append(q)
        cols = list(ccols[keep])
        # Obstacles first (sorted among themselves), then the drone on top so it is never hidden.
        ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="none", linewidths=0, zorder=1))
        dp, dc = drone_polys(parts, rotors, pos[i], R, frac[i], spin[i], prop_r, scale=2.5, shadow=False)
        sp, sc = drone_polys(parts, rotors, pos[i], R, frac[i], spin[i], prop_r, scale=2.5, shadow=True)
        sp, sc = sp[len(dp):], sc[len(dc):]
        ax.add_collection3d(Poly3DCollection(sp, facecolors=sc, edgecolors="none", zorder=2))
        ax.add_collection3d(Poly3DCollection(dp, facecolors=dc, edgecolors=(0, 0, 0, 0.3), linewidths=0.25,
                                             zorder=4))
        m = (path[:, 0] > x0) & (path[:, 0] < x1)
        ax.plot(*path[m].T, color="#35d6ff", lw=1.0, alpha=0.8, ls=(0, (4, 3)), zorder=3)
        j0 = max(0, i - int(6.0 / dt))
        tr = pos[j0:i + 1]
        tr = tr[tr[:, 0] > x0]
        if len(tr):
            ax.plot(*tr.T, color="#ffc53d", lw=1.8, zorder=3)
        ax.text2D(0.03, 0.05, "drone drawn 2.5× size in this view · dashed = planned path", transform=ax.transAxes,
                  color=DIM, fontsize=7)

        # ---------------------------------------------------------------- chase camera (true scale)
        axc.cla()
        style3d(axc)
        c, r = pos[i], 0.36
        axc.set_xlim(c[0] - r, c[0] + r), axc.set_ylim(c[1] - r, c[1] + r), axc.set_zlim(c[2] - r, c[2] + r)
        axc.set_box_aspect((1, 1, 1))
        axc.set_xticklabels([]), axc.set_yticklabels([]), axc.set_zticklabels([])
        axc.view_init(elev=20, azim=-58 + 8 * np.sin(t[i] / 5.0))
        dp, dc = drone_polys(parts, rotors, c, R, frac[i], spin[i], prop_r, scale=1.0, shadow=False)
        axc.add_collection3d(Poly3DCollection(dp, facecolors=dc, edgecolors=(0, 0, 0, 0.35), linewidths=0.3))
        clr = L["clearance"][i]
        axc.set_title(f"chase camera (true scale)  ·  red = front  ·  clearance {clr:.2f} m", color=DIM,
                      fontsize=8, pad=0)

        # ---------------------------------------------------------------- attitude strip
        axa.cla()
        axa.set_facecolor(PANEL)
        j0 = max(0, i - int(5.0 / dt))
        tt = t[j0:i + 1]
        axa.plot(tt, euler[j0:i + 1, 0], color="#4ea3ff", lw=1.3, label="roll (true)")
        axa.plot(tt, est[j0:i + 1, 0], color="#4ea3ff", lw=1.0, ls=":", label="roll (MPU-6050 est.)")
        axa.plot(tt, euler[j0:i + 1, 1], color="#ffc53d", lw=1.3, label="pitch (true)")
        axa.plot(tt, est[j0:i + 1, 1], color="#ffc53d", lw=1.0, ls=":", label="pitch (MPU-6050 est.)")
        axa.set_xlim(max(0, t[i] - 5.0), max(5.0, t[i]))
        axa.set_ylim(-90, 90)
        axa.axhline(0, color=DIM, lw=0.5)
        axa.tick_params(colors=DIM, labelsize=7)
        for s in axa.spines.values():
            s.set_color("#2a323d")
        axa.set_ylabel("deg", color=DIM, fontsize=7)
        axa.set_xlabel("time [s]", color=DIM, fontsize=7)
        axa.legend(fontsize=6, loc="upper right", ncol=2, facecolor=PANEL, edgecolor="#2a323d", labelcolor=FG)

        # ---------------------------------------------------------------- minimap + HUD
        trail_m.set_data(pos[:i + 1:3, 0], pos[:i + 1:3, 1])
        dot_m.set_data([pos[i, 0]], [pos[i, 1]])
        raw = L["imu_raw"][i]
        spd = np.linalg.norm(L["vel"][i])
        bars = " ".join(f"M{m} {'█' * int(round(frac[i, m] * 8)):<8s}{frac[i, m]*100:3.0f}%" for m in range(4))
        hud.set_text(
            f"t {t[i]:5.2f} s  alt {pos[i, 2]:4.2f} m  speed {spd:4.2f} m/s  progress {L['progress'][i]*100:3.0f}%\n"
            f"roll {euler[i, 0]:6.1f}°  pitch {euler[i, 1]:6.1f}°  clearance {clr:4.2f} m\n"
            f"{bars}\n"
            f"MPU-6050 ACCEL {raw[0]:6d} {raw[1]:6d} {raw[2]:6d}  TEMP {raw[3] / 340 + 36.53:4.1f}°C\n"
            f"         GYRO  {raw[4]:6d} {raw[5]:6d} {raw[6]:6d}  (raw LSB)")
        msg = section_name(pos[i, 0])
        if not a.no_throw and t[i] < 1.6:
            msg = "THROWN at 70° roll, spinning: self-balancing"
        if i >= len(t) - int(1.5 / dt) and not L["crashed"]:
            msg = f"COURSE COMPLETE: {L['path_length']:.1f} m, {t[-1]:.1f} s, 0 collisions"
        banner.set_text(msg)

        fig.canvas.draw()
        img = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        if writer is None:
            from PIL import Image
            Image.fromarray(img).save(out.with_name(f"{out.stem}_t{times[f]:05.1f}.png"))
        else:
            writer.send(img)
        if f % 60 == 0:
            print(f"  rendered {f}/{n_frames} frames", flush=True)
    if writer is not None:
        writer.close()
        print(f"video written to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
