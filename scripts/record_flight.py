#!/usr/bin/env python3
"""Fly the trained model and record the flight as a 3D video (MP4 or GIF).

    python scripts/record_flight.py --policy models/f450_policy.pt
    python scripts/record_flight.py --policy models/f450_policy.pt --throw --wind 2 --push 11.5
    python scripts/record_flight.py --controller pid --out runs/pid_flight.mp4
    python scripts/record_flight.py --policy models/f450_policy.pt --set payload=0.25 --out runs/heavy.gif

The video shows a wide 3D scene with the flight path and waypoints, a close-up chase camera
at true scale, true vs MPU-6050-estimated attitude, motor outputs and the raw MPU-6050 registers.
"""
import argparse
from pathlib import Path

import numpy as np
from common import make_controller, make_model

from drone_sim import DroneEnv, EnvConfig
from drone_sim.math3d import quat_to_rotmat
from drone_sim.mission import fly_mission

BG, PANEL, FG, DIM = "#0f1419", "#161c24", "#e6e6e6", "#7a8491"
FRONT, REAR, PATH, WP, WP_DONE, TGT = "#ff5a4e", "#4ea3ff", "#ffc53d", "#8a95a3", "#3ecf8e", "#ff3df2"


def drone_geometry(model):
    """Body-frame line segments and rotor centres used for drawing."""
    com = model.mass_properties().com
    rotors = model.rotor_positions - com
    r_prop = float(model.rotor_cfg.get("propeller_radius", 0.127))
    ang = np.linspace(0, 2 * np.pi, 25)
    disc = np.stack((np.cos(ang), np.sin(ang), np.zeros_like(ang)), 1) * r_prop
    body = np.array([[0.07, 0.05, 0.02], [-0.07, 0.05, 0.02], [-0.07, -0.05, 0.02], [0.07, -0.05, 0.02],
                     [0.07, 0.05, 0.02]]) - com
    return rotors, disc, body


def draw_drone(ax, pos, R, geom, rotor_frac, spin, scale=1.0, lw=2.2, thrust_arrow=True):
    rotors, disc, body = geom
    w = lambda p: pos + scale * (p @ R.T)  # noqa: E731
    centre = w(np.zeros(3))
    for i, rp in enumerate(rotors):
        c = FRONT if rp[0] > 0 else REAR
        a = w(rp)
        ax.plot(*np.c_[centre, a], color=c, lw=lw, solid_capstyle="round")
        d = w(rp + disc)
        ax.plot(*d.T, color=c, lw=0.9, alpha=0.9)
        # A spinning propeller blade: angle advances with the rotor speed.
        th = spin[i]
        blade = np.array([[np.cos(th), np.sin(th), 0], [-np.cos(th), -np.sin(th), 0]]) * disc[0, 0]
        ax.plot(*w(rp + blade).T, color=FG, lw=1.4, alpha=0.8)
        # Motor output as a vertical bar above each rotor.
        top = w(rp + [0, 0, 0.02 + 0.12 * rotor_frac[i]])
        ax.plot(*np.c_[a, top], color=c, lw=4, alpha=0.55)
    ax.plot(*w(body).T, color=FG, lw=1.4)
    if thrust_arrow:
        up = w(np.array([0, 0, 0.25]))
        ax.plot(*np.c_[centre, up], color="#3ecf8e", lw=1.2, ls="--")


def style3d(ax):
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.09, 0.11, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0.07)
        axis.label.set_color(DIM)
        axis.set_tick_params(colors=DIM, labelsize=7)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="models/f450_policy.pt")
    p.add_argument("--controller", choices=["nn", "pid"], default="nn")
    p.add_argument("--config")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--waypoints", default="0,0,1.5; 1.5,0,2; 1.5,1.5,2.5; -1,1.5,1.5; -1,-1,2; 0,0,1.5")
    p.add_argument("--hold", type=float, default=1.5)
    p.add_argument("--wind", type=float, default=2.0, help="steady wind along +x [m/s]")
    p.add_argument("--throw", action="store_true", help="release it tumbling at 70° roll")
    p.add_argument("--push", type=float, action="append", default=[], help="time [s] of a sideways shove")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", default="runs/flight_3d.mp4")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    policy = None
    if a.controller == "nn":
        from drone_sim.rl import Policy
        policy = Policy.load(a.policy)
    model = make_model(a, policy)
    over = dict(randomize=False, waypoint_change_prob=0.0, episode_seconds=1e6, max_distance=50.0)
    cfg = policy.env_cfg(**over) if policy else EnvConfig.from_dict(over)
    env = DroneEnv(1, model, cfg, seed=a.seed)
    act, _ = make_controller(a.controller, env, policy)
    wps = np.array([[float(v) for v in w.split(",")] for w in a.waypoints.split(";")])
    pushes = [(t, (0.0, 12.0, 2.0)) for t in a.push]
    print(f"flying mission with the {a.controller.upper()} controller ...")
    L = fly_mission(env, act, wps, a.hold, 8.0, wind=(a.wind, 0, 0), throw=a.throw, pushes=pushes)
    render(L, model, a)


def render(L, model, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geom = drone_geometry(model)
    t, pos, quat = L["t"], L["pos"], L["quat"]
    wps, dt = L["waypoints"], L["dt"]
    max_speed = model.max_rotor_speed
    rotor_frac = (L["rotor"] / max_speed) ** 2          # thrust fraction per motor
    # Visual propeller angle (slowed down so the eye can follow it).
    spin_angle = np.cumsum(L["rotor"] * model.spin_dirs * dt * 0.02, axis=0)
    n_frames = int(t[-1] * a.fps)
    idx = np.minimum((np.arange(n_frames) / a.fps / dt).astype(int), len(t) - 1)

    lo = np.minimum(wps.min(0), pos.min(0)) - 0.6
    hi = np.maximum(wps.max(0), pos.max(0)) + 0.6
    lo[2] = 0.0
    dpi = 100
    fig = plt.figure(figsize=(a.width / dpi, a.height / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0.0, 0.17, 0.62, 0.74], projection="3d")
    axc = fig.add_axes([0.62, 0.40, 0.38, 0.55], projection="3d")
    axa = fig.add_axes([0.665, 0.07, 0.315, 0.27])
    title = f"Neural-network autopilot  ·  {model.name}" if a.controller == "nn" else \
        f"PID autopilot  ·  {model.name}"
    fig.text(0.012, 0.965, title, color=FG, fontsize=13, weight="bold", va="center")
    mp = model.mass_properties()
    fig.text(0.012, 0.93, f"mass {mp.mass*1000:.0f} g   ·   thrust/weight {model.thrust_to_weight():.2f}   ·   "
             f"wind {np.linalg.norm(L['wind']):.1f} m/s   ·   sensors: simulated MPU-6050 + position fix",
             color=DIM, fontsize=8.5, va="center")
    hud = fig.text(0.015, 0.02, "", color=FG, fontsize=8.5, family="monospace", va="bottom")
    banner = fig.text(0.31, 0.885, "", color="#ffcf4d", fontsize=12, weight="bold", ha="center")

    import imageio_ffmpeg
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_gif = []
    writer = None
    if out.suffix.lower() == ".mp4":
        writer = imageio_ffmpeg.write_frames(str(out), (a.width, a.height), fps=a.fps, quality=8,
                                             macro_block_size=8)
        writer.send(None)

    euler, est = np.rad2deg(L["euler"]), np.rad2deg(L["est_euler"])
    events = L["events"]
    for f, i in enumerate(idx):
        R = quat_to_rotmat(quat[i])
        # ---------------------------------------------------------------- wide scene
        ax.cla()
        style3d(ax)
        ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.set_xlabel("x [m]", fontsize=8), ax.set_ylabel("y [m]", fontsize=8), ax.set_zlabel("z [m]", fontsize=8)
        ax.view_init(elev=24, azim=-60 + 12 * np.sin(t[i] / 6.0))
        k = L["wp_index"][i]
        for j, w in enumerate(wps):
            col = WP_DONE if j < k else (TGT if j == k else WP)
            ax.scatter(*w, color=col, s=70 if j == k else 30, depthshade=False, edgecolors="none")
            ax.plot([w[0], w[0]], [w[1], w[1]], [0, w[2]], color=col, lw=0.6, alpha=0.35)
            ax.text(w[0], w[1], w[2] + 0.12, str(j + 1), color=col, fontsize=8, ha="center")
        j0 = max(0, i - int(6.0 / dt))
        ax.plot(*pos[:i + 1].T, color=PATH, lw=0.6, alpha=0.25)
        ax.plot(*pos[j0:i + 1].T, color=PATH, lw=1.6)
        ax.plot(pos[j0:i + 1, 0], pos[j0:i + 1, 1], 0 * pos[j0:i + 1, 2], color="k", lw=1.0, alpha=0.5)
        ax.scatter(pos[i, 0], pos[i, 1], 0, color="k", s=60, alpha=0.5, depthshade=False)
        draw_drone(ax, pos[i], R, geom, rotor_frac[i], spin_angle[i], scale=1.8, lw=2.0, thrust_arrow=False)
        push = L["push"][i]
        if np.linalg.norm(push) > 0:
            v = push / np.linalg.norm(push) * 0.8
            s = pos[i] - v * 1.25
            e = s + v
            ax.plot(*np.c_[s, e], color="#ff3d3d", lw=3.5)
            ax.scatter(*e, color="#ff3d3d", s=90, marker="o", depthshade=False)
            ax.text(*(s - 0.1 * v), "shove", color="#ff3d3d", fontsize=9, weight="bold")
        ax.text2D(0.02, 0.04, "drone drawn 1.8× size in this view", transform=ax.transAxes, color=DIM, fontsize=7)

        # ---------------------------------------------------------------- chase camera
        axc.cla()
        style3d(axc)
        c, r = pos[i], 0.38
        axc.set_xlim(c[0] - r, c[0] + r), axc.set_ylim(c[1] - r, c[1] + r), axc.set_zlim(c[2] - r, c[2] + r)
        axc.set_box_aspect((1, 1, 1))
        axc.set_xticklabels([]), axc.set_yticklabels([]), axc.set_zticklabels([])
        axc.view_init(elev=18, azim=-60 + 12 * np.sin(t[i] / 6.0))
        draw_drone(axc, c, R, geom, rotor_frac[i], spin_angle[i], scale=1.0, lw=3.0)
        axc.plot([c[0] - r, c[0] + r], [c[1], c[1]], [c[2], c[2]], color=DIM, lw=0.5, alpha=0.4)
        axc.plot([c[0], c[0]], [c[1] - r, c[1] + r], [c[2], c[2]], color=DIM, lw=0.5, alpha=0.4)
        axc.set_title("chase camera (true scale) · red = front · bars = motor thrust", color=DIM, fontsize=8,
                      pad=0)

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
        axa.set_ylim(-80, 80)
        axa.axhline(0, color=DIM, lw=0.5)
        axa.tick_params(colors=DIM, labelsize=7)
        for s in axa.spines.values():
            s.set_color("#2a323d")
        axa.set_ylabel("deg", color=DIM, fontsize=7)
        axa.set_xlabel("time [s]", color=DIM, fontsize=7)
        axa.legend(fontsize=6, loc="upper right", ncol=2, facecolor=PANEL, edgecolor="#2a323d", labelcolor=FG)

        # ---------------------------------------------------------------- HUD
        raw = L["imu_raw"][i]
        dist = np.linalg.norm(pos[i] - L["target"][i])
        spd = np.linalg.norm(L["vel"][i])
        bars = "  ".join(f"M{m} {'█' * int(round(rotor_frac[i, m] * 10)):<10s}{rotor_frac[i, m]*100:3.0f}%"
                         for m in range(4))
        hud.set_text(
            f"t {t[i]:5.2f} s   alt {pos[i, 2]:4.2f} m   speed {spd:4.2f} m/s   "
            f"waypoint {min(k + 1, len(wps))}/{len(wps)}  dist {dist:4.2f} m\n"
            f"roll {euler[i, 0]:6.1f}°  pitch {euler[i, 1]:6.1f}°  yaw {euler[i, 2]:6.1f}°\n"
            f"{bars}\n"
            f"MPU-6050  ACCEL_X {raw[0]:6d}  ACCEL_Y {raw[1]:6d}  ACCEL_Z {raw[2]:6d}  "
            f"TEMP {raw[3] / 340 + 36.53:5.1f}°C\n"
            f"          GYRO_X  {raw[4]:6d}  GYRO_Y  {raw[5]:6d}  GYRO_Z  {raw[6]:6d}   (raw LSB)")
        msg = ""
        if a.throw and t[i] < 1.5:
            msg = "THROWN at 70° roll, spinning: self-balancing"
        for te, text in events:
            if te <= t[i] < te + 1.2:
                msg = text.upper()
        if np.linalg.norm(L["push"][max(0, i - 80):i + 1], axis=1).max() > 0:
            msg = "SHOVED SIDEWAYS: recovering"
        banner.set_text(msg)

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        if writer is not None:
            writer.send(np.ascontiguousarray(img))
        else:
            frames_gif.append(img[::2, ::2].copy())
        if f % 60 == 0:
            print(f"  rendered {f}/{n_frames} frames")
    if writer is not None:
        writer.close()
    else:
        from PIL import Image
        ims = [Image.fromarray(x) for x in frames_gif[::2]]
        ims[0].save(out, save_all=True, append_images=ims[1:], duration=int(2000 / a.fps), loop=0)
    plt.close(fig)
    print(f"video written to {out}  ({n_frames} frames, {t[-1]:.1f} s)")


if __name__ == "__main__":
    main()
