# Dimension-Wise Importance Sampling Weight Clipping (PyTorch)

PyTorch implementation of [Dimension-Wise Importance Sampling Weight Clipping
for Sample-Efficient Reinforcement Learning](https://arxiv.org/abs/1905.02363)
(Han & Sung, ICML 2019), ported from the authors' TensorFlow 1 implementation,
which was built on [OpenAI Baselines](https://github.com/openai/baselines).

```bibtex
@article{han2019dimension,
  title={Dimension-Wise Importance Sampling Weight Clipping for Sample-Efficient Reinforcement Learning},
  author={Han, Seungyul and Sung, Youngchul},
  journal={arXiv preprint arXiv:1905.02363},
  year={2019}
}
```

## Installation

```sh
poetry install          # default PyPI wheels (CUDA build on Linux)
poetry install -E cpu   # CPU-only wheels, no CUDA libraries
```

Requires Poetry 2.5 or newer. Training runs on the CPU unless `--device cuda`
is passed, so the `cpu` extra is enough for the defaults.

## Training

```sh
poetry run disc-train --env Humanoid-v5 --num_timesteps 1e7 --log_dir ./Results/Humanoid-v5
```

Options mirror the original TF implementation's `run_disc.py` (`--leng`, `--epsilon`,
`--epsilon_b`, `--jtarg`, `--gaev`, `--seed`, `--num_timesteps`,
`--log_dir`). Per-update statistics are written to `<log_dir>/seed<seed>/` as
TensorBoard scalars (indexed by total timesteps) and as `progress.csv` using the
same keys as the original; nothing is printed to stdout. View them with

```sh
poetry run tensorboard --logdir ./Results
```

Additional options: `--no_eval` skips the deterministic evaluation (10
episodes after every update, which costs more environment steps than training
itself on long-horizon tasks), `--save_interval N` saves checkpoints, and
`--load_path` resumes from one.

## Recording videos

Train with `--save_interval N` so checkpoints are written, then:

```sh
poetry run disc-record Results/Hopper-v5/seed1/checkpoints/00500.pt --episodes 3
```

This writes one MP4 per episode to `./videos` using the deterministic (mean)
action. Rendering is offscreen through EGL, so it works on headless machines;
set `MUJOCO_GL` to override. `--live` opens a window instead of recording.
The environment is read from the checkpoint; pass `--env` for checkpoints
that predate this.

## Differences from the original TF implementation

- Environments are Gymnasium MuJoCo (`*-v5`) instead of gym 0.15 / mujoco-py
  `*-v2`. Dynamics and observation spaces differ slightly, so returns are not
  directly comparable to the paper's.
- Time-limit truncation is treated as termination, as in the original.
- Monitor CSV files (`0.monitor.csv`) are not written; episode statistics are
  logged to `progress.csv` instead.
- Checkpoints also store the observation/return normalization statistics.

## Tests

```sh
poetry run pytest
```

`tests/test_algorithm.py` checks the loss against a numpy transliteration of
the original TF graph, including gradients. The original code is in this
repository's git history (before the port replaced it).
