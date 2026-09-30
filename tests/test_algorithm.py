"""Checks the torch port against a numpy transliteration of the TF1 graph."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest
import torch

from disc_torch.algorithm import (
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

EPSILON, ALPHA_IS, VF_COEF = 0.4, 0.7, 0.5


def reference_loss(
    inputs: dict[str, np.ndarray], r_denominator: float | None = None
) -> tuple[float, np.ndarray]:
    """Numpy version of the Model graph in the original TF code.

    The original was ``baselines/DISC/DISC.py``, now only in git history.

    ``r_denominator`` fixes the stop-gradient term for finite differences.
    """
    a, mean, logstd = inputs["actions"], inputs["mean"], inputs["logstd"]
    old_mean, old_logstd = inputs["old_mean"], inputs["old_logstd"]
    adv, ret, vpred = inputs["advs"], inputs["returns"], inputs["vpred"]
    old_v, on = inputs["old_values"], inputs["on_policy"]

    neglogpac = gaussian_neglogp(a, mean, logstd)
    entropy = np.mean(np.sum(logstd + 0.5 * np.log(2 * np.pi * np.e), axis=-1))
    vclip = old_v + np.clip(vpred - old_v, -EPSILON, EPSILON)
    vf_loss = 0.5 * np.mean(
        np.maximum(np.square(vpred - ret), np.square(vclip - ret))
    )
    ratio = dimensionwise_ratio(a, mean, logstd, old_mean, old_logstd)
    sgn = np.ones_like(ratio) * np.sign(adv)[:, None]
    ratio_clip = np.clip(ratio, 1 - EPSILON, 1 + EPSILON)
    r = np.prod(sgn * np.minimum(ratio * sgn, ratio_clip * sgn), axis=-1)
    denom = np.mean(r) if r_denominator is None else r_denominator
    j_disc = np.mean(-r * adv / denom)
    is_loss = 0.5 * np.mean(np.square(neglogpac - inputs["old_neglogp"]) * on)
    kl = np.mean(
        np.sum(
            logstd
            - old_logstd
            + 0.5
            * (np.exp(old_logstd) ** 2 + np.square(mean - old_mean))
            / np.exp(logstd) ** 2
            - 0.5,
            axis=1,
        )
        * on
    )
    j_disc = j_disc + ALPHA_IS * is_loss
    loss = j_disc + vf_loss * VF_COEF
    return loss, np.array([j_disc, vf_loss, entropy, is_loss, kl])


def random_inputs(seed: int, n: int = 32, d: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    old_mean = rng.normal(size=(n, d))
    old_logstd = np.tile(rng.normal(scale=0.3, size=d), (n, 1))
    actions = old_mean + np.exp(old_logstd) * rng.normal(size=(n, d))
    on_policy = np.zeros(n)
    on_policy[n // 2 :] = 2.0
    return {
        "actions": actions,
        "mean": old_mean + rng.normal(scale=0.1, size=(n, d)),
        "logstd": old_logstd + rng.normal(scale=0.05, size=(n, d)),
        "old_mean": old_mean,
        "old_logstd": old_logstd,
        "old_neglogp": gaussian_neglogp(actions, old_mean, old_logstd),
        "advs": rng.normal(size=n),
        "returns": rng.normal(size=n),
        "vpred": rng.normal(size=n),
        "old_values": rng.normal(size=n),
        "on_policy": on_policy,
    }


def torch_loss(
    inputs: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    t = {k: torch.tensor(v, dtype=torch.float64) for k, v in inputs.items()}
    for k in ("mean", "logstd", "vpred"):
        t[k].requires_grad_(True)
    batch = Batch(
        obs=torch.empty(0),
        returns=t["returns"],
        advs=t["advs"],
        actions=t["actions"],
        old_values=t["old_values"],
        old_neglogp=t["old_neglogp"],
        old_mean=t["old_mean"],
        old_logstd=t["old_logstd"],
        on_policy=t["on_policy"],
    )
    loss, stats = disc_loss(
        t["mean"], t["logstd"], t["vpred"], batch, EPSILON, ALPHA_IS, VF_COEF
    )
    return loss, stats, t


@pytest.mark.parametrize("seed", range(5))
def test_loss_matches_reference(seed: int) -> None:
    inputs = random_inputs(seed)
    ref_loss, ref_stats = reference_loss(inputs)
    loss, stats, _ = torch_loss(inputs)
    np.testing.assert_allclose(loss.item(), ref_loss, rtol=1e-10)
    np.testing.assert_allclose(stats.numpy(), ref_stats, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_gradient_treats_mean_r_as_constant(seed: int) -> None:
    inputs = random_inputs(seed)
    loss, _, t = torch_loss(inputs)
    loss.backward()
    _, ref_stats = reference_loss(inputs)
    ratio = dimensionwise_ratio(
        inputs["actions"],
        inputs["mean"],
        inputs["logstd"],
        inputs["old_mean"],
        inputs["old_logstd"],
    )
    sgn = np.sign(inputs["advs"])[:, None]
    clipped = np.clip(ratio, 1 - EPSILON, 1 + EPSILON)
    r_mean = np.mean(np.prod(sgn * np.minimum(ratio * sgn, clipped * sgn), -1))
    assert np.isfinite(ref_stats).all()

    h = 1e-6
    for key in ("mean", "logstd", "vpred"):
        grad = t[key].grad
        assert grad is not None
        flat = inputs[key].reshape(-1)
        for i in range(0, flat.size, max(1, flat.size // 7)):
            plus, minus = dict(inputs), dict(inputs)
            plus[key] = flat.copy()
            minus[key] = flat.copy()
            plus[key][i] += h
            minus[key][i] -= h
            plus[key] = plus[key].reshape(inputs[key].shape)
            minus[key] = minus[key].reshape(inputs[key].shape)
            numeric = (
                reference_loss(plus, r_mean)[0]
                - reference_loss(minus, r_mean)[0]
            ) / (2 * h)
            assert grad.reshape(-1)[i].item() == pytest.approx(
                numeric, rel=1e-5, abs=1e-7
            )


def test_rho_is_product_of_dimensionwise_ratios() -> None:
    inputs = random_inputs(0)
    rho = np.exp(
        inputs["old_neglogp"]
        - gaussian_neglogp(inputs["actions"], inputs["mean"], inputs["logstd"])
    )
    rho_dim = dimensionwise_ratio(
        inputs["actions"],
        inputs["mean"],
        inputs["logstd"],
        inputs["old_mean"],
        inputs["old_logstd"],
    )
    np.testing.assert_allclose(rho, rho_dim.prod(-1), rtol=1e-10)


def test_gae_hand_computed() -> None:
    rew = np.array([1.0, 2.0, 3.0])
    done = np.array([False, False, True])  # obs 2 starts a new episode
    values = np.array([0.5, 1.0, 1.5, 2.0])
    gamma, lam = 0.9, 0.8
    ret, adv = gae(rew, done, values, gamma, lam)
    d2 = 3.0 + gamma * 2.0 - 1.5
    d1 = 2.0 - 1.0  # obs 1 is terminal: no bootstrap
    d0 = 1.0 + gamma * 1.0 - 0.5
    expected = np.array([d0 + gamma * lam * d1, d1, d2])
    np.testing.assert_allclose(adv, expected, rtol=1e-6)
    np.testing.assert_allclose(ret, expected + values[:-1], rtol=1e-6)


def test_gae_v_reduces_to_gae_when_on_policy() -> None:
    rng = np.random.default_rng(0)
    rew = rng.normal(size=50)
    done = rng.random(50) < 0.1
    values = rng.normal(size=51)
    rho = 1.0 + rng.random(50)  # truncated to 1 everywhere
    ret_v, adv_v = gae_v(rew, done, values, rho, 0.99, 0.95)
    ret, adv = gae(rew, done, values, 0.99, 0.95)
    np.testing.assert_allclose(adv_v, adv, rtol=1e-5)
    np.testing.assert_allclose(ret_v, ret, rtol=1e-5)


def test_gae_v_truncates_traces() -> None:
    rew = np.array([0.0, 1.0])
    done = np.zeros(2, bool)
    values = np.zeros(3)
    rho = np.array([1.0, 0.0])  # zero weight cuts the trace after step 1
    ret, adv = gae_v(rew, done, values, rho, 1.0, 1.0)
    np.testing.assert_allclose(adv, [0.0, 1.0])
    np.testing.assert_allclose(ret, [0.0, 0.0])


def test_batch_inclusion_mask() -> None:
    rho_dim = np.ones((6, 2))
    rho_dim[2:4] = 1.5  # second batch deviates by 0.5 > epsilon_b
    np.testing.assert_array_equal(
        batch_inclusion_mask(rho_dim, 2, 0.1), [1, 1, 0, 0, 1, 1]
    )


def test_normalize_advantages_zero_weighted_mean() -> None:
    rng = np.random.default_rng(0)
    adv = rng.normal(size=100) + 3.0
    rho = rng.random(100) * 2
    out = normalize_advantages(torch.tensor(adv), torch.tensor(rho)).numpy()
    trunc_rho = np.minimum(1.0, rho)
    assert np.mean(trunc_rho * out) == pytest.approx(0.0, abs=1e-10)
    # Scale uses the population std, as np.std in the original.
    expected = (adv - np.mean(trunc_rho * adv) / trunc_rho.mean()) / (
        np.std(trunc_rho * adv) + 1e-8
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_append_to_buffer_keeps_newest() -> None:
    buf = None
    for i in range(4):
        seg = {"x": np.full(3, i), "y": np.full((3, 2), i)}
        buf = append_to_buffer(buf, seg, 6)
    assert buf is not None
    np.testing.assert_array_equal(buf["x"], [2, 2, 2, 3, 3, 3])
    assert buf["y"].shape == (6, 2)
