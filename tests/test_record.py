"""End-to-end: train briefly, save a checkpoint, record a video of it."""

# pylint: disable=missing-function-docstring
from pathlib import Path

import gymnasium as gym
import torch

from disc_torch.logger import Logger
from disc_torch.record import main
from disc_torch.train import Config, learn


def test_record_from_checkpoint(tmp_path: Path) -> None:
    config = Config(
        total_timesteps=64 * 2,
        nsteps=64,
        gradstepsperepoch=4,
        noptepochs=1,
        save_interval=2,
    )
    logger = Logger(tmp_path)
    learn(gym.make("InvertedPendulum-v5"), config, logger=logger)
    logger.close()
    checkpoint = tmp_path / "checkpoints" / "00002.pt"
    assert torch.load(checkpoint, weights_only=False)["env_id"] == (
        "InvertedPendulum-v5"
    )

    out = tmp_path / "videos"
    main([str(checkpoint), "--out", str(out), "--episodes", "2"])
    videos = sorted(out.glob("*.mp4"))
    assert [v.name for v in videos] == [
        "InvertedPendulum-v5-00002-episode-0.mp4",
        "InvertedPendulum-v5-00002-episode-1.mp4",
    ]
    assert all(v.stat().st_size > 0 for v in videos)
