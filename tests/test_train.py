"""Network initialization and an end-to-end smoke run."""

# pylint: disable=missing-function-docstring

import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from disc_torch.logger import Logger
from disc_torch.networks import ActorCritic
from disc_torch.running_mean_std import RunningMeanStd
from disc_torch.train import Config, learn


def test_actor_critic_shapes_and_init() -> None:
    model = ActorCritic(obs_dim=5, act_dim=3)
    mean, logstd, value = model(torch.zeros(7, 5))
    assert mean.shape == logstd.shape == (7, 3)
    assert value.shape == (7,)
    assert torch.all(logstd == 0)
    assert torch.all(model.pi_mean.weight.abs() < 0.02)
    layer = model.pi_torso[2]
    assert isinstance(layer, torch.nn.Linear)
    w = layer.weight
    torch.testing.assert_close(w @ w.T, 2 * torch.eye(64), atol=1e-5, rtol=0)


def test_running_mean_std_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(loc=3.0, scale=2.0, size=(1000, 4))
    rms = RunningMeanStd(shape=(4,))
    for chunk in np.split(data, 10):
        rms.update(chunk)
    np.testing.assert_allclose(rms.mean, data.mean(0), rtol=1e-5)
    np.testing.assert_allclose(rms.var, data.var(0), rtol=1e-4)


def cuda_works() -> bool:
    """True if CUDA is present and has kernels for the installed GPU."""
    if not torch.cuda.is_available():
        return False
    try:
        (torch.ones(1, device="cuda") + 1).item()
    except RuntimeError:
        return False
    return True


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not cuda_works(), reason="no usable GPU"),
        ),
    ],
)
def test_learn_smoke(tmp_path: Path, device: str) -> None:
    env = gym.make("InvertedPendulum-v5")
    eval_env = gym.make("InvertedPendulum-v5")
    config = Config(
        total_timesteps=64 * 6,
        nsteps=64,
        gradstepsperepoch=4,
        noptepochs=2,
        replay_length=3,
        save_interval=3,
    )
    logger = Logger(tmp_path)
    model = learn(env, config, eval_env, logger, device=device)
    assert all(p.device.type == "cpu" for p in model.parameters())
    logger.close()

    with open(tmp_path / "progress.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert all(np.isfinite(float(r["policy_loss"])) for r in rows)
    assert int(rows[-1]["total_timesteps"]) == 64 * 6
    assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir()) == [
        "00001.pt",
        "00003.pt",
        "00006.pt",
    ]


def test_logger_writes_tensorboard(tmp_path: Path) -> None:
    logger = Logger(tmp_path)
    for step in (64, 128):
        logger.logkv("policy_loss", step / 64)
        logger.dumpkvs(step)
    logger.close()

    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    events = accumulator.Scalars("policy_loss")
    assert [(e.step, e.value) for e in events] == [(64, 1.0), (128, 2.0)]
