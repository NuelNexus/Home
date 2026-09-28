# Beginner's Guide — How to Run This Project

This guide assumes you know very little about Python or the command line. Every step is spelled out.
If you get stuck, check the **Troubleshooting** section near the bottom — it covers the most common problems.

---

## What does this project actually do?

It's a flight simulator for a toy drone, plus an AI "brain" (a small neural network) that has learned to
fly it by itself — balancing, recovering if thrown, and dodging obstacles — using only fake sensor data,
the same way a real drone would. You can:

- Watch the AI fly the drone, as a video, with sound.
- Change how heavy parts of the drone are (a bigger battery, an added camera, etc.) and see how it copes.
- Train your own AI pilot from scratch.

No real drone, no real sensors, and no graphics card are needed. Everything runs on an ordinary laptop.

---

## Before you start

You need:
1. **Python**, version 3.9 or newer. ([python.org/downloads](https://www.python.org/downloads/) if you
   don't have it — on Windows, tick "Add Python to PATH" during install.)
2. **About 1 GB of free disk space.** Almost all of that is one library (PyTorch), the tool that runs the
   AI brain.
3. A **graphics card is not required.** This is designed to run entirely on a normal CPU.

Open a terminal (**Terminal** on Mac/Linux, **Command Prompt** or **PowerShell** on Windows) and check
Python is installed:

```bash
python3 --version
```

If that says something like `Python 3.11.2`, you're good. If it says "command not found", try `python`
instead of `python3` everywhere in this guide (Windows usually uses plain `python`).

---

## Step 1 — Get the project onto your computer

If you already have the folder (e.g. you downloaded a ZIP or cloned it with `git`), open a terminal
**inside that folder**. On Mac/Linux/Windows, that's usually:

```bash
cd path/to/the/folder
```

(`cd` means "change directory" — it moves your terminal into that folder. Drag the folder onto the
terminal window to auto-fill its path on Mac, or type it manually.)

---

## Step 2 — (Recommended) Make a clean, isolated Python space

This step keeps this project's libraries separate from anything else on your computer, so nothing clashes.
It's optional but strongly recommended.

```bash
python3 -m venv venv
```

This creates a folder called `venv` containing a private copy of Python just for this project. Now
"turn it on" (this is called *activating* it):

- **Mac / Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Windows (Command Prompt):**
  ```bash
  venv\Scripts\activate.bat
  ```
- **Windows (PowerShell):**
  ```bash
  venv\Scripts\Activate.ps1
  ```

You'll know it worked because your terminal prompt now starts with `(venv)`. You'll need to repeat this
"activate" step every time you open a new terminal window to work on this project (you only need to run
`python3 -m venv venv` once, ever).

---

## Step 3 — Install the required libraries

This is the one command that installs everything the project needs:

```bash
pip install -r requirements.txt
```

What this does: it reads the file `requirements.txt` and downloads each library it lists. The biggest one
by far is **PyTorch**, the library that runs the neural network — expect this step to take a few minutes
and download roughly 300–400 MB. `requirements.txt` is set up to fetch the small **CPU-only** version of
PyTorch (about 200 MB) rather than the version built for gaming graphics cards, which would otherwise be
about 2 GB. You don't need to do anything extra for this — it's automatic — just don't separately run
`pip install torch` on its own, since that would fetch the big one.

When it finishes without red "ERROR" text, you're ready.

---

## Step 4 — Fly the drone that's already trained (no training needed)

A trained AI pilot is already included, so you can try flying immediately:

```bash
python3 scripts/fly_mission.py --policy models/f450_policy.pt
```

This makes the drone fly through 6 invisible checkpoints (waypoints) in mid-air. You'll see text like:

```
flying 6 waypoints with the NN controller
  waypoint 0 [0.  0.  1.5] -> reached in 4.1s, error 0.181 m
  ...
```

That's it working. It also saves a flight log and a still image into a `runs/mission` folder so you can
look at the flight path afterwards.

Try adding `--throw` to make it start by being released mid-air while tumbling, to see it catch itself:

```bash
python3 scripts/fly_mission.py --policy models/f450_policy.pt --throw
```

---

## Step 5 — Watch it as a proper video, with an obstacle course and sound

This is the main event. It plans a safe route through a set of obstacles (poles, walls with windows, a
low bar to duck under, a slalom), then has the AI fly it, and renders the whole thing as a video with a
soundtrack generated from the simulated motor noise.

```bash
python3 scripts/record_obstacles.py
```

This takes a few minutes (there's no graphics card doing the work, just the CPU, drawing every single
frame of video). When it's done, you'll find the video at:

```
docs/obstacle_flight.mp4
```

Double-click it to watch it in any video player.

**Useful options** (add these after the command, e.g.
`python3 scripts/record_obstacles.py --wind 2`):

| Option | What it does |
|---|---|
| `--wind 2` | Add a stronger sideways wind (in metres/second) |
| `--set payload=0.2` | Strap on a 200-gram payload and watch it cope |
| `--no-audio` | Skip generating the soundtrack (a bit faster) |
| `--audio-only` | Add/redo just the sound on a video you already rendered (fast, ~1 minute) |
| `--preview 1,10,20` | Instead of a full video, just save a few still images at those times (seconds) — much faster, good for checking things quickly |
| `--out my_video.mp4` | Save under a different name |

If you'd rather have a plain flight (no obstacle course) with a shove partway through:

```bash
python3 scripts/record_flight.py --throw --wind 2 --push 11.5
```
This saves to `docs/flight_3d.mp4`.

---

## Step 6 — Change how heavy the drone's parts are

Every part of the drone (battery, camera, flight computer, etc.) has a weight you can adjust. First, see
the current breakdown:

```bash
python3 scripts/components.py
```

This prints a table of every part, its weight, and where it's positioned, plus how well the drone would
fly (its thrust-to-weight ratio).

Now try changing something:

```bash
python3 scripts/components.py --set battery=0.25
```

That sets the battery to 250 grams (0.25 kg). You can combine changes and even add new parts:

```bash
python3 scripts/components.py --set battery=0.25 --set payload=0.1 --add camera=0.045@0.08,0,-0.01
```

To actually **fly** with a changed drone, pass the same `--set` option to any of the flying/video scripts:

```bash
python3 scripts/record_obstacles.py --set battery=0.25 --set payload=0.1
```

The AI pilot was trained to cope with weight changes like this within reason, so it should still fly, just
a little differently.

---

## Step 7 — Check how good the AI pilot really is

This flies 200 randomised test flights and reports how often it crashes and how accurate it is, compared
to a simpler backup autopilot:

```bash
python3 scripts/evaluate.py --policy models/f450_policy.pt
```

---

## Step 8 — (Advanced) Train your own AI pilot from scratch

This is optional — a trained pilot is already included — but you can train a new one, e.g. for a drone
you've customised with `--set` in Step 6.

```bash
python3 scripts/train.py --steps 30e6 --out runs/my_drone
```

What's happening: the software flies hundreds of virtual copies of the drone at once, over and over,
slowly getting better each time (this is called *reinforcement learning* — trial and error, but very
fast and all in software). `--steps 30e6` means "30 million practice attempts", which takes roughly an
hour on a normal 4-core laptop CPU. You'll see a running scoreboard printed every few seconds, e.g.:

```
it   100 | steps   1.64M | return    779.4 | ... | crash  13.0% | final dist  1.46 m | ...
```

The `crash` percentage should go down and `final dist` (how close it gets to its target) should go down
too, as training progresses. When it's done, your new pilot is saved as `runs/my_drone/best.pt`. Fly it
with:

```bash
python3 scripts/record_obstacles.py --policy runs/my_drone/best.pt
```

Want to check on it sooner? You don't have to wait for the full 30 million steps — press `Ctrl+C` to stop
early, and `runs/my_drone/last.pt` will still have whatever progress was saved so far.

---

## Step 9 — (Very advanced) Export the AI brain for real hardware

If you want to run the trained AI on an actual microcontroller (ESP32, STM32, Arduino) next to a real
MPU-6050 sensor chip, this converts it into a plain C file with no external dependencies:

```bash
python3 scripts/export_policy.py models/f450_policy.pt --out my_export/
```

This is only useful if you're building real hardware — read the main `README.md` for the safety caveats
before flying anything for real.

---

## Troubleshooting

**"python3: command not found" or "'python3' is not recognized"**
Try `python` instead of `python3` in every command above. On Windows this is usually the correct one.

**"No module named 'torch'" (or numpy / matplotlib / etc.)**
The install in Step 3 didn't finish properly, or you're in a different terminal than the one where you
activated `venv`. Re-run:
```bash
pip install -r requirements.txt
```
Make sure your terminal prompt shows `(venv)` at the start, if you set up a virtual environment.

**The video-recording script fails with an error about ffmpeg or matplotlib**
Those are in `requirements.txt` too, but if something went wrong, install them directly:
```bash
pip install matplotlib pillow imageio-ffmpeg
```

**Rendering a video takes a really long time**
That's normal — the video is being drawn frame by frame by the CPU (no graphics card is used). A
25–30 second flight typically takes 5–10 minutes to render into an MP4. Use `--preview 1,10,20`
(Step 5) if you just want a quick look rather than a full video.

**`pip install torch` downloaded 2 GB instead of ~200 MB**
Don't run `pip install torch` by itself — always use `pip install -r requirements.txt`, which is
configured to fetch the small CPU-only version automatically.

**I want to start over / free up disk space**
Delete the `venv` folder (`rm -rf venv` on Mac/Linux, delete it in File Explorer on Windows) and repeat
Steps 2–3.

---

## Quick command reference

| I want to... | Command |
|---|---|
| Install everything | `pip install -r requirements.txt` |
| See the drone's part weights | `python3 scripts/components.py` |
| Change a part's weight | `python3 scripts/components.py --set battery=0.25` |
| Fly the pre-trained AI (text only) | `python3 scripts/fly_mission.py --policy models/f450_policy.pt` |
| Record a video: obstacle course + sound | `python3 scripts/record_obstacles.py` |
| Record a video: simple flight + shove | `python3 scripts/record_flight.py --throw --wind 2 --push 11.5` |
| Score the AI pilot (200 test flights) | `python3 scripts/evaluate.py --policy models/f450_policy.pt` |
| Train a brand-new AI pilot | `python3 scripts/train.py --steps 30e6 --out runs/my_drone` |
| Run the automated tests | `python3 -m pytest -q tests` |
| Export the brain for real hardware | `python3 scripts/export_policy.py models/f450_policy.pt --out my_export/` |

For the full technical explanation of how the simulator, sensors and AI training work, see
[`README.md`](README.md).
