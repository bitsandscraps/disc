"""DISC objective, return estimators and replay-buffer selection.

Everything here is a pure function so it can be tested in isolation. Numpy is
used for the per-iteration buffer computations (as in the original), and torch
only for the differentiable loss.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch

LOSS_NAMES = ("policy_loss", "value_loss", "policy_entropy", "ISloss", "KLloss")


def gaussian_neglogp(
    x: np.ndarray, mean: np.ndarray, logstd: np.ndarray
) -> np.ndarray:
    """Negative log-density of a diagonal Gaussian, summed over last axis."""
    return (
        0.5 * np.sum(np.square((x - mean) / np.exp(logstd)), axis=-1)
        + 0.5 * np.log(2.0 * np.pi) * x.shape[-1]
        + np.sum(logstd, axis=-1)
    )


def dimensionwise_ratio(
    x: np.ndarray,
    mean: np.ndarray,
    logstd: np.ndarray,
    old_mean: np.ndarray,
    old_logstd: np.ndarray,
) -> np.ndarray:
    """Per-dimension importance weights pi(x_d) / pi_old(x_d)."""
    return np.exp(
        -0.5 * np.square((x - mean) / np.exp(logstd))
        - logstd
        + 0.5 * np.square((x - old_mean) / np.exp(old_logstd))
        + old_logstd
    )


def gae(
    rew: np.ndarray,
    done: np.ndarray,
    values: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """TD(lambda) value targets and GAE advantages.

    ``done[t]`` marks that observation ``t`` starts a new episode, and
    ``values`` has one more entry than ``rew`` (the bootstrap value).
    """
    done = np.append(done, 0)
    horizon = len(rew)
    adv = np.empty(horizon, np.float32)
    lastgaelam = 0.0
    for t in reversed(range(horizon)):
        nonterminal = 1 - done[t + 1]
        delta = rew[t] + gamma * values[t + 1] * nonterminal - values[t]
        adv[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    return adv + values[:-1], adv


def gae_v(
    rew: np.ndarray,
    done: np.ndarray,
    values: np.ndarray,
    rho: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """V-trace value targets and GAE-V advantages (truncated IS weights)."""
    done = np.append(done, 0)
    trunc_rho = np.minimum(1.0, np.append(rho, 1.0))
    horizon = len(rew)
    adv = np.empty(horizon, np.float32)
    lastgaelam = 0.0
    for t in reversed(range(horizon)):
        nonterminal = 1 - done[t + 1]
        delta = rew[t] + gamma * values[t + 1] * nonterminal - values[t]
        adv[t] = delta + gamma * lam * nonterminal * lastgaelam
        lastgaelam = trunc_rho[t] * adv[t]
    return trunc_rho[:-1] * adv + values[:-1], adv


def batch_inclusion_mask(
    rho_dim: np.ndarray, nsteps: int, epsilon_b: float
) -> np.ndarray:
    """1 for samples in stored batches whose mean |rho_d - 1| <= epsilon_b."""
    mask = np.zeros(len(rho_dim))
    for i in range(len(rho_dim) // nsteps):
        chunk = slice(i * nsteps, (i + 1) * nsteps)
        condition = np.mean(np.abs(rho_dim[chunk] - 1.0) + 1.0)
        mask[chunk] = float(condition <= 1 + epsilon_b)
    return mask


def normalize_advantages(adv: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Normalize so that the mean of min(1, rho) * A is zero (as in IMPALA)."""
    trunc_rho = rho.clamp(max=1.0)
    radv = trunc_rho * adv
    return (adv - radv.mean() / trunc_rho.mean()) / (
        radv.std(correction=0) + 1e-8
    )


def append_to_buffer(
    buffer: dict[str, np.ndarray] | None,
    seg: dict[str, np.ndarray],
    max_len: int,
) -> dict[str, np.ndarray]:
    """Append a rollout to the replay buffer, keeping the newest ``max_len``."""
    if buffer is None:
        return seg
    return {k: np.concatenate([buffer[k], seg[k]])[-max_len:] for k in seg}


@dataclass
class Batch:
    """A training minibatch; advantages are already normalized."""

    obs: torch.Tensor
    returns: torch.Tensor
    advs: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_neglogp: torch.Tensor
    old_mean: torch.Tensor
    old_logstd: torch.Tensor
    on_policy: torch.Tensor


def disc_loss(
    mean: torch.Tensor,
    logstd: torch.Tensor,
    vpred: torch.Tensor,
    batch: Batch,
    epsilon: float,
    alpha_is: float,
    vf_coef: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Total DISC loss and the stats named in ``LOSS_NAMES``.

    ``policy_loss`` includes the adaptive IS penalty, matching the original.
    """
    std = logstd.exp()
    old_std = batch.old_logstd.exp()
    act_dim = batch.actions.shape[-1]
    neglogp = (
        0.5 * ((batch.actions - mean) / std).square().sum(-1)
        + 0.5 * math.log(2.0 * math.pi) * act_dim
        + logstd.sum(-1)
    )
    entropy = (logstd + 0.5 * math.log(2.0 * math.pi * math.e)).sum(-1).mean()

    # Clipped value loss (the clip range is shared with the policy).
    vpred_clipped = batch.old_values + (vpred - batch.old_values).clamp(
        -epsilon, epsilon
    )
    vf_loss = (
        0.5
        * torch.maximum(
            (vpred - batch.returns).square(),
            (vpred_clipped - batch.returns).square(),
        ).mean()
    )

    # Dimension-wise clipped surrogate.
    ratio = torch.exp(
        -0.5 * ((batch.actions - mean) / std).square()
        - logstd
        + 0.5 * ((batch.actions - batch.old_mean) / old_std).square()
        + batch.old_logstd
    )
    sgn = batch.advs.sign().unsqueeze(1)
    ratio_clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    r = (sgn * torch.minimum(ratio * sgn, ratio_clipped * sgn)).prod(-1)
    surrogate = (-r * batch.advs / r.mean().detach()).mean()

    # IS and KL penalties are computed on on-policy samples only.
    is_loss = (
        0.5 * ((neglogp - batch.old_neglogp).square() * batch.on_policy).mean()
    )
    kl = (
        (
            logstd
            - batch.old_logstd
            + 0.5
            * (old_std.square() + (mean - batch.old_mean).square())
            / std.square()
            - 0.5
        ).sum(-1)
        * batch.on_policy
    ).mean()

    policy_loss = surrogate + alpha_is * is_loss
    loss = policy_loss + vf_coef * vf_loss
    stats = torch.stack([policy_loss, vf_loss, entropy, is_loss, kl]).detach()
    return loss, stats
