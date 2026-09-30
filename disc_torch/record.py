"""Record MP4 videos (or show a live window) of a trained DISC policy."""

import argparse
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch

from disc_torch.networks import ActorCritic
from disc_torch.runners import normalize_obs, policy_forward
from disc_torch.running_mean_std import RunningMeanStd


def load_policy(
    checkpoint: dict[str, Any], env: gym.Env[Any, Any]
) -> tuple[ActorCritic, RunningMeanStd]:
    """Rebuild the model and observation filter saved by ``disc-train``."""
    assert env.observation_space.shape is not None
    assert env.action_space.shape is not None
    model = ActorCritic(
        env.observation_space.shape[0], env.action_space.shape[0]
    )
    model.load_state_dict(checkpoint["model"])
    ob_rms = RunningMeanStd(shape=env.observation_space.shape)
    ob_rms.load_state_dict(checkpoint["ob_rms"])
    return model, ob_rms


def rollout(
    env: gym.Env[Any, Any],
    model: ActorCritic,
    ob_rms: RunningMeanStd,
    seed: int,
) -> tuple[float, int]:
    """Run one deterministic (mean-action) episode; return (return, length)."""
    obs, _ = env.reset(seed=seed)
    ret, length, done = 0.0, 0, False
    while not done:
        action, _ = policy_forward(model, normalize_obs(obs, ob_rms))
        obs, reward, terminated, truncated, _ = env.step(action)
        ret += float(reward)
        length += 1
        done = bool(terminated or truncated)
    return ret, length


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", help="checkpoint saved by disc-train")
    parser.add_argument(
        "--env", help="environment ID (default: the one in the checkpoint)"
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="video directory", default="videos")
    parser.add_argument(
        "--live", help="show a window instead of recording", action="store_true"
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``disc-record``."""
    args = parse_args(argv)
    if not args.live:
        # Offscreen rendering works headless and doesn't need a display.
        os.environ.setdefault("MUJOCO_GL", "egl")

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    env_id = args.env or checkpoint.get("env_id")
    if env_id is None:
        raise SystemExit("Checkpoint has no env ID; pass --env.")

    render_kwargs = {"width": args.width, "height": args.height}
    if args.live:
        env = gym.make(env_id, render_mode="human", **render_kwargs)
    else:
        env = RecordVideo(
            gym.make(env_id, render_mode="rgb_array", **render_kwargs),
            video_folder=args.out,
            name_prefix=f"{env_id}-{Path(args.checkpoint).stem}",
            episode_trigger=lambda _: True,
        )
    model, ob_rms = load_policy(checkpoint, env)

    try:
        for i in range(args.episodes):
            ret, length = rollout(env, model, ob_rms, args.seed + i)
            print(f"episode {i}: return {ret:.1f}, length {length}")
    finally:
        env.close()
    if not args.live:
        print(f"videos written to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
