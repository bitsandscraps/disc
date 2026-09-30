"""DISC training loop and command-line entry point."""

import argparse
from collections import deque
import copy
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from disc_torch.algorithm import (
    LOSS_NAMES,
    Batch,
    append_to_buffer,
    batch_inclusion_mask,
    dimensionwise_ratio,
    disc_loss,
    gae,
    gae_v,
    gaussian_neglogp,
    normalize_advantages,
)
from disc_torch.logger import Logger
from disc_torch.networks import ActorCritic
from disc_torch.runners import EvalRunner, Runner


@dataclass(frozen=True)
class Config:
    """DISC hyperparameters (defaults follow the paper and original code).

    nsteps: size of a sample batch (N).
    lr: initial learning rate, decayed linearly and floored at ``min_lr``.
    gradstepsperepoch: minibatches per epoch; minibatch size is nsteps / this.
    epsilon: clipping factor for dimension-wise clipping.
    replay_length: maximum number of sample batches in the buffer (L).
    j_targ: IS target constant.
    epsilon_b: batch inclusion factor.
    gaev: use GAE-V (V-trace) instead of GAE.
    """

    total_timesteps: int = 3_000_000
    nsteps: int = 2048
    lr: float = 3e-4
    min_lr: float = 1e-4
    vf_coef: float = 0.5
    gamma: float = 0.99
    lam: float = 0.95
    gradstepsperepoch: int = 32
    noptepochs: int = 10
    epsilon: float = 0.4
    replay_length: int = 64
    j_targ: float = 0.001
    epsilon_b: float = 0.1
    gaev: bool = True
    evaluate: bool = True
    log_interval: int = 1
    save_interval: int = 0
    seed: int = 1


def to_tensor(
    x: np.ndarray, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Convert a numpy array to a float32 tensor on ``device``."""
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def train_step(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    minibatch: dict[str, torch.Tensor],
    epsilon: float,
    alpha_is: float,
    vf_coef: float,
) -> torch.Tensor:
    """One gradient step; returns the ``LOSS_NAMES`` stats on the device."""
    batch = Batch(
        obs=minibatch["ob"],
        returns=minibatch["ret"],
        advs=normalize_advantages(minibatch["adv"], minibatch["rho"]),
        actions=minibatch["ac"],
        old_values=minibatch["value"],
        old_neglogp=minibatch["neglogp"],
        old_mean=minibatch["mean"],
        old_logstd=minibatch["logstd"],
        on_policy=minibatch["on_policy"],
    )
    mean, logstd, vpred = model(batch.obs)
    loss, stats = disc_loss(
        mean, logstd, vpred, batch, epsilon, alpha_is, vf_coef
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return stats


def safemean(xs: list[float]) -> float:
    """Mean that returns NaN instead of warning on an empty list."""
    return float(np.mean(xs)) if xs else float("nan")


def learn(
    env: gym.Env[Any, Any],
    config: Config,
    eval_env: gym.Env[Any, Any] | None = None,
    logger: Logger | None = None,
    load_path: str | None = None,
    device: torch.device | str = "cpu",
) -> ActorCritic:
    """Train a DISC agent on ``env`` and return the (CPU) model.

    Rollouts always use a CPU copy of the model, since single-observation
    inference is faster there; gradient steps and the per-update buffer
    re-evaluation run on ``device``. The copy is synced after every update.
    """
    cfg = config
    logger = logger or Logger(None)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    assert env.observation_space.shape is not None
    assert env.action_space.shape is not None
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"Observation space dimension : {obs_dim}")
    print(f"Action space dimension : {act_dim}")

    device = torch.device(device)
    act_model = ActorCritic(obs_dim, act_dim)
    runner = Runner(env, act_model, cfg.nsteps, cfg.gamma, cfg.seed)
    if load_path is not None:
        checkpoint = torch.load(
            load_path, map_location="cpu", weights_only=False
        )
        act_model.load_state_dict(checkpoint["model"])
        runner.ob_rms.load_state_dict(checkpoint["ob_rms"])
        runner.ret_rms.load_state_dict(checkpoint["ret_rms"])
    model = act_model
    if device.type != "cpu":
        model = copy.deepcopy(act_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    eval_runner = None
    if eval_env is not None and cfg.evaluate:
        eval_runner = EvalRunner(
            eval_env, act_model, runner.obfilt, 10 * cfg.nsteps, cfg.seed
        )

    nsteps = cfg.nsteps
    nbatch_train = nsteps // cfg.gradstepsperepoch
    assert nsteps % cfg.gradstepsperepoch == 0
    nupdates = cfg.total_timesteps // nsteps
    epinfobuf: deque[dict[str, float]] = deque(maxlen=10)
    eval_epinfos: list[dict[str, float]] = []
    buffer: dict[str, np.ndarray] | None = None
    alpha_is = 1.0
    tfirststart = time.time()

    for update in range(1, nupdates + 1):
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = max(cfg.min_lr, cfg.lr * frac)
        for group in optimizer.param_groups:
            group["lr"] = lrnow

        seg, final_obs, final_done, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        if eval_runner is not None:
            eval_epinfos = eval_runner.run()
        buffer = append_to_buffer(buffer, seg, cfg.replay_length * nsteps)
        size = len(buffer["ob"])

        # Re-evaluate the whole buffer under the current policy and filters.
        ob = runner.obfilt(buffer["ob"])
        ob_final = np.concatenate([ob, runner.obfilt(final_obs[None])])
        with torch.no_grad():
            mean_t, logstd_t, values_t = model(to_tensor(ob_final, device))
        values = values_t.cpu().numpy()
        values[-1] *= 1.0 - final_done
        mean_now = mean_t[:-1].cpu().numpy().astype(np.float64)
        logstd_now = logstd_t[:-1].detach().cpu().numpy().astype(np.float64)
        rho = np.exp(
            buffer["neglogp"]
            - gaussian_neglogp(buffer["ac"], mean_now, logstd_now)
        )

        rew = runner.rewfilt(buffer["rew"])
        if cfg.gaev:
            ret, adv = gae_v(
                rew, buffer["done"], values, rho, cfg.gamma, cfg.lam
            )
        else:
            ret, adv = gae(rew, buffer["done"], values, cfg.gamma, cfg.lam)

        rho_dim = dimensionwise_ratio(
            buffer["ac"], mean_now, logstd_now, buffer["mean"], buffer["logstd"]
        )
        prior_prob = batch_inclusion_mask(rho_dim, nsteps, cfg.epsilon_b)

        inds_on = np.arange(size - nsteps, size)
        inds_off = np.arange(size - nsteps)
        nbatch_off = int((prior_prob.sum() - nsteps) / nsteps * nbatch_train)
        # The IS penalty is averaged over the whole minibatch, so on-policy
        # samples are up-weighted by the number of included batches.
        on_policy = np.zeros(size)
        on_policy[-nsteps:] = prior_prob.sum() / nsteps

        arrays_np = {
            "ob": ob,
            "ret": ret,
            "adv": adv,
            "ac": buffer["ac"],
            "value": values[:-1],
            "neglogp": buffer["neglogp"],
            "mean": buffer["mean"],
            "logstd": buffer["logstd"],
            "on_policy": on_policy,
            "rho": rho,
        }
        arrays = {k: to_tensor(v, device) for k, v in arrays_np.items()}
        mblossvals = []
        for _ in range(cfg.noptepochs):
            for _ in range(nsteps // nbatch_train):
                idx_on = rng.choice(inds_on, nbatch_train)
                if nbatch_off > 0:
                    p_off = prior_prob[:-nsteps] / prior_prob[:-nsteps].sum()
                    idx_off = rng.choice(inds_off, nbatch_off, p=p_off)
                    idx = np.concatenate([idx_off, idx_on])
                else:
                    idx = idx_on
                idx_t = torch.as_tensor(idx, device=device)
                minibatch = {k: v[idx_t] for k, v in arrays.items()}
                mblossvals.append(
                    train_step(
                        model,
                        optimizer,
                        minibatch,
                        cfg.epsilon,
                        alpha_is,
                        cfg.vf_coef,
                    )
                )
        lossvals = torch.stack(mblossvals).mean(0).cpu().numpy()
        if model is not act_model:
            act_model.load_state_dict(model.state_dict())

        # Adapt the IS penalty coefficient towards the target J_targ.
        is_loss = lossvals[LOSS_NAMES.index("ISloss")]
        if is_loss > cfg.j_targ * 1.5:
            alpha_is *= 2
        elif is_loss < cfg.j_targ / 1.5:
            alpha_is /= 2
        alpha_is = float(np.clip(alpha_is, 2**-10, 64))

        tnow = time.time()
        if update % cfg.log_interval == 0 or update == 1:
            logger.logkv("adaptive IS loss factor", alpha_is)
            logger.logkv("clipping factor", cfg.epsilon)
            logger.logkv("learning rate", lrnow)
            logger.logkv("included batches", int(prior_prob.sum()) // nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update * nsteps)
            logger.logkv("fps", int(nsteps / (tnow - tstart)))
            logger.logkv("eprewmean", safemean([e["r"] for e in epinfobuf]))
            logger.logkv("eplenmean", safemean([e["l"] for e in epinfobuf]))
            if eval_runner is not None:
                logger.logkv(
                    "eval_eprewmean", safemean([e["r"] for e in eval_epinfos])
                )
                logger.logkv(
                    "eval_eplenmean", safemean([e["l"] for e in eval_epinfos])
                )
            logger.logkv("time_elapsed", tnow - tfirststart)
            for name, value in zip(LOSS_NAMES, lossvals):
                logger.logkv(name, float(value))
            logger.dumpkvs()

        if (
            cfg.save_interval
            and (update % cfg.save_interval == 0 or update == 1)
            and logger.log_dir is not None
        ):
            checkdir = logger.log_dir / "checkpoints"
            checkdir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": act_model.state_dict(),
                    "ob_rms": runner.ob_rms.state_dict(),
                    "ret_rms": runner.ret_rms.state_dict(),
                    "env_id": env.spec.id if env.spec is not None else None,
                },
                checkdir / f"{update:05d}.pt",
            )
    return act_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line (flags mirror the original run_disc.py)."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--env", help="environment ID", default="Ant-v5")
    parser.add_argument("--leng", help="replay length", type=int, default=64)
    parser.add_argument(
        "--epsilon", help="clipping factor", type=float, default=0.4
    )
    parser.add_argument(
        "--epsilon_b", help="batch inclusion factor", type=float, default=0.1
    )
    parser.add_argument(
        "--jtarg", help="IS target constant", type=float, default=0.001
    )
    parser.add_argument("--gaev", help="use GAE-V", type=int, default=1)
    parser.add_argument("--seed", help="random seed", type=int, default=1)
    parser.add_argument("--num_timesteps", type=float, default=3e6)
    parser.add_argument("--log_dir", help="log dir", default=None)
    parser.add_argument(
        "--no_eval", help="skip deterministic evaluation", action="store_true"
    )
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--load_path", default=None)
    parser.add_argument(
        "--device", help="device for gradient steps (cpu, cuda)", default="cpu"
    )
    parser.add_argument(
        "--torch_threads",
        help="CPU threads; match the number of physical cores",
        type=int,
        default=4,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``disc-train``."""
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    config = Config(
        total_timesteps=int(args.num_timesteps),
        epsilon=args.epsilon,
        replay_length=args.leng,
        j_targ=args.jtarg,
        epsilon_b=args.epsilon_b,
        gaev=bool(args.gaev),
        evaluate=not args.no_eval,
        save_interval=args.save_interval,
        seed=args.seed,
    )
    log_dir = None
    if args.log_dir is not None:
        log_dir = Path(args.log_dir) / f"seed{args.seed}"
    logger = Logger(log_dir)
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    print(f"Training DISC on {args.env} with {config}")
    try:
        learn(env, config, eval_env, logger, args.load_path, args.device)
    finally:
        env.close()
        eval_env.close()
        logger.close()


if __name__ == "__main__":
    main()
