"""Rollout collection on a single Gymnasium environment."""

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from disc_torch.algorithm import gaussian_neglogp
from disc_torch.networks import ActorCritic
from disc_torch.running_mean_std import RunningMeanStd

EpisodeInfo = dict[str, float]


class EpisodeTracker:
    """Accumulates undiscounted return and length (like baselines' Monitor)."""

    def __init__(self) -> None:
        self.ret = 0.0
        self.length = 0

    def step(self, reward: float, done: bool) -> EpisodeInfo | None:
        """Record a step; return the episode info when ``done``."""
        self.ret += reward
        self.length += 1
        if not done:
            return None
        info = {"r": self.ret, "l": self.length}
        self.ret, self.length = 0.0, 0
        return info


def normalize_obs(
    obs: np.ndarray,
    rms: RunningMeanStd,
    clip: float = 10.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize observations with running statistics and clip."""
    return np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip)


def policy_forward(
    model: ActorCritic, obs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Action mean and log-std for a single (already filtered) observation."""
    with torch.no_grad():
        mean, logstd, _ = model(torch.as_tensor(obs, dtype=torch.float32))
    return mean.numpy().astype(np.float64), logstd.detach().numpy().astype(
        np.float64
    )


class Runner:
    """Collects ``nsteps`` transitions and maintains the normalization filters.

    Observations are stored unfiltered; the buffer is re-filtered with the
    latest statistics at every update, as in the original implementation.
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        model: ActorCritic,
        nsteps: int,
        gamma: float,
        seed: int,
        clipob: float = 10.0,
        cliprew: float = 10.0,
    ) -> None:
        self.env = env
        self.model = model
        self.nsteps = nsteps
        self.gamma = gamma
        self.clipob = clipob
        self.cliprew = cliprew
        self.eps = 1e-8
        self.obs, _ = env.reset(seed=seed)
        self.done = False
        self.ret = 0.0
        self.episode = EpisodeTracker()
        assert env.observation_space.shape is not None
        self.ob_rms = RunningMeanStd(shape=env.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=())

    def obfilt(self, obs: np.ndarray) -> np.ndarray:
        """Normalize and clip observations with the running statistics."""
        return normalize_obs(obs, self.ob_rms, self.clipob, self.eps)

    def rewfilt(self, rews: np.ndarray) -> np.ndarray:
        """Scale rewards by the running std of discounted returns and clip."""
        return np.clip(
            rews / np.sqrt(self.ret_rms.var + self.eps),
            -self.cliprew,
            self.cliprew,
        )

    def run(
        self,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, bool, list[EpisodeInfo]]:
        """Return the rollout, the final observation/done flag and episodes.

        ``done[t]`` marks that ``ob[t]`` is the first observation of a new
        episode (baselines' convention); time-limit truncation counts as done.
        """
        keys = ("ob", "rew", "done", "ac", "neglogp", "mean", "logstd")
        mb: dict[str, list[Any]] = {k: [] for k in keys}
        epinfos = []
        for _ in range(self.nsteps):
            mean, logstd = policy_forward(self.model, self.obfilt(self.obs))
            noise = torch.randn(mean.shape, dtype=torch.float64).numpy()
            action = mean + np.exp(logstd) * noise
            mb["ob"].append(self.obs.copy())
            mb["ac"].append(action)
            mb["neglogp"].append(gaussian_neglogp(action, mean, logstd))
            mb["done"].append(self.done)
            mb["mean"].append(mean)
            mb["logstd"].append(logstd)

            obs, reward, terminated, truncated, _ = self.env.step(action)
            self.done = bool(terminated or truncated)
            if self.done:
                obs, _ = self.env.reset()
            self.obs = obs
            self.ob_rms.update(self.obs[None])
            self.ret = self.ret * self.gamma + float(reward)
            self.ret_rms.update(np.array([self.ret]))
            if self.done:
                self.ret = 0.0

            epinfo = self.episode.step(float(reward), self.done)
            if epinfo is not None:
                epinfos.append(epinfo)
            mb["rew"].append(reward)

        seg = {
            "ob": np.asarray(mb["ob"]),
            "rew": np.asarray(mb["rew"], np.float32),
            "done": np.asarray(mb["done"], np.bool_),
            "ac": np.asarray(mb["ac"]),
            "neglogp": np.asarray(mb["neglogp"], np.float32),
            "mean": np.asarray(mb["mean"]),
            "logstd": np.asarray(mb["logstd"]),
        }
        return seg, self.obs, self.done, epinfos


class EvalRunner:
    """Deterministic (mean-action) evaluation for up to ``num_episodes``.

    Like the original, the environment state persists between calls, so the
    first episode of a call may have started during the previous one.
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        model: ActorCritic,
        obfilt: Callable[[np.ndarray], np.ndarray],
        max_steps: int,
        seed: int,
        num_episodes: int = 10,
    ) -> None:
        self.env = env
        self.model = model
        self.obfilt = obfilt
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.obs, _ = env.reset(seed=seed)
        self.episode = EpisodeTracker()

    def run(self) -> list[EpisodeInfo]:
        """Run until ``num_episodes`` end or ``max_steps`` elapse."""
        epinfos = []
        for _ in range(self.max_steps):
            action, _ = policy_forward(self.model, self.obfilt(self.obs))
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)
            if done:
                obs, _ = self.env.reset()
            self.obs = obs
            epinfo = self.episode.step(float(reward), done)
            if epinfo is not None:
                epinfos.append(epinfo)
                if len(epinfos) == self.num_episodes:
                    break
        return epinfos
