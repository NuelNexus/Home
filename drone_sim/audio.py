"""Physically driven drone audio, synthesised from a flight log.

Every sound follows the simulation:

* propeller buzz: each rotor is a harmonic voice at the blade-pass frequency
  f_bp = n_blades · Ω / 2π (about 130-190 Hz for the F450 at hover), with loudness
  rising with thrust (∝ Ω²)
* motor whine: brushless commutation tone at pole_pairs · Ω / 2π
* prop wash: broadband noise whose level grows with Ω³ (aerodynamic noise power)
* wind rush: band-passed noise that grows with airspeed²
* whooshes: short filtered-noise swells when the drone passes close to an obstacle

Stereo: rotors on the left of the airframe (+y) lean left, right rotors lean right.
"""
from __future__ import annotations

import wave

import numpy as np


def _bandpass_noise(n, lo, hi, sr, rng):
    """White noise band-limited to [lo, hi] Hz via FFT masking (fast, zero phase)."""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    X *= (f >= lo) & (f <= hi)
    y = np.fft.irfft(X, n)
    return y / (np.std(y) + 1e-12)


def synthesize(L: dict, rotor_positions, max_rotor_speed: float, sr: int = 44100, n_blades: int = 2,
               pole_pairs: int = 7, duration: float | None = None, seed: int = 0, obstacle_clearance=None):
    """Return a (N, 2) float32 stereo signal in [-1, 1] for the flight log ``L``."""
    rng = np.random.default_rng(seed)
    t_log = np.concatenate(([0.0], np.asarray(L["t"])))
    rotor = np.vstack((L["rotor"][:1], L["rotor"]))                 # rad/s, (T, 4)
    vel = np.vstack((L["vel"][:1], L["vel"]))
    T = duration or float(t_log[-1])
    n = int(T * sr)
    ts = np.arange(n) / sr
    Om = np.stack([np.interp(ts, t_log, rotor[:, k]) for k in range(rotor.shape[1])], 1)
    speed = np.interp(ts, t_log, np.linalg.norm(vel, axis=1))
    frac = np.clip(Om / max_rotor_speed, 0, 1)

    left = np.zeros(n)
    right = np.zeros(n)
    pan = np.clip(0.5 + 1.5 * np.asarray(rotor_positions)[:, 1], 0.15, 0.85)    # +y (left) -> left channel
    harm_amp = np.array([1.0, 0.55, 0.38, 0.22, 0.15, 0.1, 0.07, 0.05])
    for k in range(Om.shape[1]):
        # Tiny per-rotor speed wobble (blade imbalance / ESC jitter) keeps the four voices from phasing.
        wobble = 1 + 0.004 * np.sin(2 * np.pi * (3.1 + 0.7 * k) * ts + k)
        f_bp = n_blades * Om[:, k] * wobble / (2 * np.pi)
        phase = 2 * np.pi * np.cumsum(f_bp) / sr + rng.uniform(0, 2 * np.pi)
        buzz = sum(a * np.sin((h + 1) * phase) for h, a in enumerate(harm_amp))
        buzz *= frac[:, k] ** 2 * 1.6
        f_whine = pole_pairs * Om[:, k] / (2 * np.pi)
        whine = 0.06 * np.sin(2 * np.pi * np.cumsum(f_whine) / sr) * frac[:, k]
        voice = buzz + whine
        left += voice * (1 - pan[k])
        right += voice * pan[k]

    # Prop wash: broadband, level ∝ (mean Ω)^3
    wash_env = np.mean(frac, axis=1) ** 3 * 2.2
    wash = _bandpass_noise(n, 250, 5000, sr, rng) * wash_env * 0.35
    # Wind rush past the airframe
    wind = _bandpass_noise(n, 80, 1200, sr, rng) * np.clip(speed / 3.0, 0, 1.5) ** 2 * 0.25
    left += wash + wind * 0.9
    right += wash * 0.95 + wind

    # Whooshes when passing close to obstacles (clearance dips below 0.6 m)
    if obstacle_clearance is not None:
        clr = np.interp(ts, t_log[1:], np.asarray(obstacle_clearance))
        close = np.clip((0.6 - clr) / 0.35, 0, 1) ** 1.5
        whoosh = _bandpass_noise(n, 300, 2500, sr, rng)
        env = np.convolve(close, np.hanning(int(0.25 * sr)) / (0.125 * sr), mode="same")
        left += whoosh * env * 0.35
        right += whoosh * env * 0.35

    out = np.stack((left, right), 1)
    out /= np.percentile(np.abs(out), 99.9) + 1e-9
    out = np.tanh(out * 0.9) * 0.89                     # gentle limiter
    fade = int(0.3 * sr)
    out[:fade] *= np.linspace(0, 1, fade)[:, None]
    out[-fade:] *= np.linspace(1, 0, fade)[:, None]
    return out.astype(np.float32)


def write_wav(path, audio: np.ndarray, sr: int = 44100) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def mux(video_path, wav_path, out_path, ffmpeg: str) -> None:
    """Attach ``wav_path`` to ``video_path`` without re-encoding the video."""
    import subprocess
    subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", str(video_path), "-i", str(wav_path),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", str(out_path)], check=True)
