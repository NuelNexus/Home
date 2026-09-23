"""Proximal Policy Optimisation (clipped objective, GAE, adaptive learning rate)."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..env import ACT_DIM, OBS_DIM, DroneEnv
from .networks import ActorCritic, RunningMeanStd


@dataclass
class PPOConfig:
    rollout_steps: int = 64
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    desired_kl: float = 0.01          # adaptive LR target (as in rsl_rl / legged_gym)
    entropy_coef: float = 0.0
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    reward_scale: float = 0.1
    actor_hidden: tuple = (64, 64)
    critic_hidden: tuple = (256, 256)


class PPO:
    def __init__(self, env: DroneEnv, cfg: PPOConfig | None = None, device: str = "cpu", log_fn=print):
        self.env, self.cfg, self.device, self.log = env, cfg or PPOConfig(), torch.device(device), log_fn
        c = self.cfg
        self.net = ActorCritic(OBS_DIM, ACT_DIM, c.actor_hidden, c.critic_hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=c.lr)
        self.lr = c.lr
        self.obs_rms = RunningMeanStd(OBS_DIM)
        self.total_steps = 0
        self.ep_returns = deque(maxlen=200)
        self.ep_lengths = deque(maxlen=200)
        self.ep_crash = deque(maxlen=200)
        self.ep_dist = deque(maxlen=200)

    def _t(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def collect(self, obs_raw):
        c, env, n, T = self.cfg, self.env, self.env.n, self.cfg.rollout_steps
        buf_obs = np.zeros((T, n, OBS_DIM), np.float32)
        buf_act = np.zeros((T, n, ACT_DIM), np.float32)
        buf_logp = np.zeros((T, n), np.float32)
        buf_rew = np.zeros((T, n), np.float32)
        buf_done = np.zeros((T, n), np.float32)
        buf_val = np.zeros((T + 1, n), np.float32)
        raw_obs = []
        for t in range(T):
            raw_obs.append(obs_raw)
            obs = self.obs_rms.normalize(obs_raw)
            ot = self._t(obs)
            dist = self.net.dist(ot)
            act = dist.sample()
            buf_obs[t], buf_act[t] = obs, act.cpu().numpy()
            buf_logp[t] = dist.log_prob(act).sum(-1).cpu().numpy()
            buf_val[t] = self.net.value(ot).cpu().numpy()
            obs_raw, rew, done, info = env.step(buf_act[t])
            rew = rew * c.reward_scale
            if done.any():
                idx = info["done_idx"]
                trunc = info["truncated"][idx]
                if trunc.any():
                    # Time-limit truncation is not a real terminal state: bootstrap.
                    v_term = self.net.value(self._t(self.obs_rms.normalize(info["terminal_obs"][trunc]))).cpu().numpy()
                    rew[idx[trunc]] += c.gamma * v_term
                self.ep_returns.extend(info["episode_return"])
                self.ep_lengths.extend(info["episode_length"])
                self.ep_crash.extend(info["crashed"][idx])
                self.ep_dist.extend(info["final_distance"])
            buf_rew[t], buf_done[t] = rew, done
        buf_val[T] = self.net.value(self._t(self.obs_rms.normalize(obs_raw))).cpu().numpy()
        self.obs_rms.update(np.concatenate(raw_obs))
        self.total_steps += T * n

        adv = np.zeros((T, n), np.float32)
        last = np.zeros(n, np.float32)
        for t in reversed(range(T)):
            nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + c.gamma * buf_val[t + 1] * nonterm - buf_val[t]
            last = delta + c.gamma * c.gae_lambda * nonterm * last
            adv[t] = last
        ret = adv + buf_val[:T]
        batch = dict(obs=buf_obs, act=buf_act, logp=buf_logp, adv=adv, ret=ret, val=buf_val[:T])
        return {k: v.reshape(T * n, *v.shape[2:]) for k, v in batch.items()}, obs_raw

    def update(self, batch):
        c = self.cfg
        B = batch["obs"].shape[0]
        data = {k: self._t(v) for k, v in batch.items()}
        adv = data["adv"]
        data["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
        mb = B // c.minibatches
        stats = []
        for _ in range(c.epochs):
            perm = torch.randperm(B, device=self.device)
            for i in range(c.minibatches):
                j = perm[i * mb:(i + 1) * mb]
                dist = self.net.dist(data["obs"][j])
                logp = dist.log_prob(data["act"][j]).sum(-1)
                ratio = (logp - data["logp"][j]).exp()
                a = data["adv"][j]
                pg = -torch.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a).mean()
                v = self.net.value(data["obs"][j])
                v_clip = data["val"][j] + (v - data["val"][j]).clamp(-c.clip, c.clip)
                vl = torch.max((v - data["ret"][j]) ** 2, (v_clip - data["ret"][j]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + c.value_coef * vl - c.entropy_coef * ent
                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - data["logp"][j])).mean().item()
                # Adaptive learning rate keeps each update close to the target KL.
                if c.desired_kl:
                    if kl > 2.0 * c.desired_kl:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif kl < 0.5 * c.desired_kl:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()
                stats.append((pg.item(), vl.item(), ent.item(), kl))
        return np.mean(stats, axis=0)

    def checkpoint(self, extra: dict | None = None) -> dict:
        ck = {
            "model": self.net.state_dict(),
            "obs_rms": self.obs_rms.state_dict(),
            "actor_hidden": list(self.cfg.actor_hidden),
            "critic_hidden": list(self.cfg.critic_hidden),
            "ppo_config": asdict(self.cfg),
            "env_config": self.env.cfg.to_dict(),
            "drone_config": self.env.model.to_config(),
            "total_steps": self.total_steps,
        }
        ck.update(extra or {})
        return ck

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ck["model"])
        self.obs_rms.load_state_dict(ck["obs_rms"])
        self.total_steps = ck.get("total_steps", 0)

    def train(self, total_steps: int, out_dir: str | Path, log_every: int = 10, save_every: int = 50):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        obs = self.env.reset()
        it, best, t0 = 0, -np.inf, time.time()
        start_steps = self.total_steps
        history = []
        while self.total_steps - start_steps < total_steps:
            batch, obs = self.collect(obs)
            pg, vl, ent, kl = self.update(batch)
            it += 1
            if self.ep_returns and it % log_every == 0:
                fps = (self.total_steps - start_steps) / (time.time() - t0)
                r, L = np.mean(self.ep_returns), np.mean(self.ep_lengths)
                crash, dist = np.mean(self.ep_crash), np.mean(self.ep_dist)
                std = self.net.log_std.exp().mean().item()
                row = dict(it=it, steps=self.total_steps, ret=r, len=L, crash=crash, dist=dist, std=std, lr=self.lr,
                           kl=kl, vloss=vl, fps=fps)
                history.append(row)
                self.log(f"it {it:5d} | steps {self.total_steps/1e6:6.2f}M | return {r:8.1f} | len {L:6.1f} | "
                         f"crash {crash*100:5.1f}% | final dist {dist:5.2f} m | std {std:.3f} | lr {self.lr:.1e} | "
                         f"{fps:6.0f} sps")
                score = r * (1.0 - crash)
                if score > best and len(self.ep_returns) >= 50:
                    best = score
                    torch.save(self.checkpoint({"score": score}), out / "best.pt")
            if it % save_every == 0:
                torch.save(self.checkpoint(), out / "last.pt")
        torch.save(self.checkpoint(), out / "last.pt")
        return history
