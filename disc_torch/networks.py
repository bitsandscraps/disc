"""Actor-critic network matching baselines' ``mlp`` + ``value_network='copy'``.

The policy and value function use separate MLPs of identical shape. The
policy is a diagonal Gaussian whose log-std is a state-independent parameter.
"""

import math

import torch
from torch import nn


def orthogonal_linear(in_dim: int, out_dim: int, scale: float) -> nn.Linear:
    """Linear layer with orthogonal weights and zero bias (baselines ``fc``)."""
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=scale)
    nn.init.zeros_(layer.bias)
    return layer


def mlp(in_dim: int, hidden: int, num_layers: int) -> nn.Sequential:
    """Tanh MLP torso with orthogonal init of scale sqrt(2)."""
    layers: list[nn.Module] = []
    for _ in range(num_layers):
        layers += [orthogonal_linear(in_dim, hidden, math.sqrt(2)), nn.Tanh()]
        in_dim = hidden
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Gaussian policy and value function with separate torsos."""

    def __init__(
        self, obs_dim: int, act_dim: int, hidden: int = 64, num_layers: int = 2
    ) -> None:
        super().__init__()
        self.pi_torso = mlp(obs_dim, hidden, num_layers)
        self.pi_mean = orthogonal_linear(hidden, act_dim, 0.01)
        self.logstd = nn.Parameter(torch.zeros(act_dim))
        self.vf_torso = mlp(obs_dim, hidden, num_layers)
        self.vf_head = orthogonal_linear(hidden, 1, 1.0)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the action mean, action log-std and state value."""
        mean = self.pi_mean(self.pi_torso(obs))
        logstd = self.logstd.expand_as(mean)
        value = self.vf_head(self.vf_torso(obs)).squeeze(-1)
        return mean, logstd, value
