"""Unit + integration tests for `code/c4_common.py`.

No Granite load here: the statistics are tested against hand-computed values and
a synthetic monotone effect surface, and the single integration test talks to the
shared daemon over HTTP (and skips when it is down).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

import c4_common as cc
from c4_common import (
    DaemonClient,
    daemon_available,
    kl_nats,
    match_effect,
    orthogonalize,
    quantile,
    signflip_p,
)


# --------------------------------------------------------------------------- #
# KL
# --------------------------------------------------------------------------- #
def test_kl_identical_is_zero():
    lg = np.array([0.3, -1.2, 4.5, 2.0], dtype=np.float32)
    assert kl_nats(lg, lg.copy()) == 0.0
    # float32 cancellation can produce ~-1e-9; it must clamp, not raise.
    assert kl_nats(lg, lg.copy()) >= 0.0


def test_kl_matches_hand_computed_value_and_is_asymmetric():
    a = np.zeros(3)  # uniform
    b = np.log(np.array([0.5, 0.25, 0.25]))  # p_b = (.5, .25, .25)

    # KL(p_b || p_a) = .5*ln(.5/(1/3)) + .25*ln(.25/(1/3)) + .25*ln(.25/(1/3))
    expect_ba = 0.5 * math.log(0.5 / (1 / 3)) + 2 * 0.25 * math.log(0.25 / (1 / 3))
    # KL(p_a || p_b) = (1/3)*ln((1/3)/.5) + (1/3)*ln((1/3)/.25) + (1/3)*ln((1/3)/.25)
    expect_ab = (1 / 3) * math.log((1 / 3) / 0.5) + 2 * (1 / 3) * math.log((1 / 3) / 0.25)

    got_ba = kl_nats(a, b)  # KL(p_b || p_a): second argument is the numerator
    got_ab = kl_nats(b, a)

    assert got_ba == pytest.approx(expect_ba, abs=1e-12)
    assert got_ab == pytest.approx(expect_ab, abs=1e-12)
    assert got_ba > 0.0 and got_ab > 0.0
    assert got_ba != pytest.approx(got_ab, abs=1e-6)  # KL is not symmetric


def test_kl_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        kl_nats(np.zeros(3), np.zeros(4))


# --------------------------------------------------------------------------- #
# quantile
# --------------------------------------------------------------------------- #
def test_quantile_raises_below_require_n():
    with pytest.raises(ValueError, match="needs n >= 22"):
        quantile(list(range(1, 11)), 0.95)  # n=10: old pct() returned the max
    with pytest.raises(ValueError, match="needs n >= 22"):
        quantile(list(range(1, 22)), 0.95)  # n=21 is the boundary
    quantile(list(range(1, 23)), 0.95)  # n=22 is accepted


def test_quantile_returns_expected_order_statistic():
    xs = list(range(1, 23))  # 1..22, n=22
    assert quantile(xs, 0.95).value == 21.0  # ceil(0.95*22) = 21
    assert quantile(xs, 0.5).value == 11.0  # ceil(0.5*22) = 11
    assert quantile(xs, 1 - 1e-9).value == 22.0

    # Order-invariant, and the maximum is only returned for q -> 1.
    shuffled = xs[::-1]
    assert quantile(shuffled, 0.95).value == 21.0


def test_quantile_ci_brackets_point_and_tightens_with_n():
    small = quantile(list(range(1, 23)), 0.5)
    assert small.lo <= small.value <= small.hi
    assert (small.lo, small.hi) == (7.0, 17.0)  # exact order statistics

    big = quantile(list(range(1, 201)), 0.5)
    assert big.lo <= big.value <= big.hi
    # Width in *probability* units shrinks with n: 10/22 vs ~19/200.
    assert (big.hi - big.lo) / big.n < (small.hi - small.lo) / small.n


# --------------------------------------------------------------------------- #
# sign-flip permutation test
# --------------------------------------------------------------------------- #
def test_signflip_all_positive_matches_two_over_two_n():
    p = signflip_p([1.0] * 10)
    assert p == pytest.approx(2 / 1024, abs=0.0)  # exact: 2 of 1024 patterns
    assert signflip_p([0.5] * 8) == pytest.approx(2 / 256, abs=0.0)


def test_signflip_all_zero_is_one():
    assert signflip_p([0.0] * 10) == 1.0


def test_signflip_is_deterministic_exact_and_monte_carlo():
    exact = [0.4, -0.1, 0.9, 0.0, 0.2, 0.7, -0.3]
    assert signflip_p(exact) == signflip_p(exact)

    many = [0.3] * 25  # n > exact_max_n -> Monte Carlo
    first = signflip_p(many)
    assert first == signflip_p(many)  # fixed seed
    assert 0.0 < first < 0.01  # all-positive, 25 pairs


def test_signflip_rejects_empty():
    with pytest.raises(ValueError):
        signflip_p([])


# --------------------------------------------------------------------------- #
# orthogonalize
# --------------------------------------------------------------------------- #
def test_orthogonalize_removes_axis_and_keeps_norm():
    rng = np.random.default_rng(11)
    for _ in range(5):
        r = rng.normal(size=64)
        v = rng.normal(size=64)
        out = orthogonalize(r, v)
        cos = float(out @ v) / (np.linalg.norm(out) * np.linalg.norm(v))
        assert abs(cos) < 1e-6
        assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(v), rel=1e-9)


def test_orthogonalize_rejects_degenerate_inputs():
    v = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-zero v"):
        orthogonalize(v, np.zeros(3))
    with pytest.raises(ValueError, match="parallel"):
        orthogonalize(3.0 * v, v)


# --------------------------------------------------------------------------- #
# match_effect (synthetic surface, no model)
# --------------------------------------------------------------------------- #
SAT = 5.0


def _fake_te(_client, _prompt, _layer, _vec, coef, _token_sets, chat=False):
    """Monotone saturating surface: f(0)=0, f(inf)=SAT, f(4)=3.16."""
    return SAT * (1.0 - math.exp(-float(coef) / 4.0))


def test_match_effect_converges_on_monotone_surface(monkeypatch):
    monkeypatch.setattr(cc, "target_te", _fake_te)
    alpha, achieved, reached = match_effect(
        object(),
        "p",
        8,
        np.zeros(4),
        2.0,
        {"comment_ids": [0], "code_ids": [1]},
        max_alpha=64.0,
        tol=0.005,
    )
    assert reached is True
    assert achieved == pytest.approx(2.0, rel=0.02)
    assert 0.0 < alpha < 64.0
    # analytic inverse: alpha = -4*ln(1 - 2/5) = 2.0433
    assert alpha == pytest.approx(-4.0 * math.log(1.0 - 2.0 / SAT), rel=0.02)


def test_match_effect_handles_negative_target(monkeypatch):
    monkeypatch.setattr(cc, "target_te", lambda *a, **k: -_fake_te(*a, **k))
    alpha, achieved, reached = match_effect(
        object(),
        "p",
        8,
        np.zeros(4),
        -1.5,
        {"comment_ids": [0], "code_ids": [1]},
        max_alpha=64.0,
        tol=0.02,
    )
    assert reached is True
    assert achieved == pytest.approx(-1.5, rel=0.02)
    assert alpha > 0.0


def test_match_effect_unreachable_returns_flag_without_raising(monkeypatch):
    monkeypatch.setattr(cc, "target_te", _fake_te)
    alpha, achieved, reached = match_effect(
        object(),
        "p",
        8,
        np.zeros(4),
        SAT + 1.0,
        {"comment_ids": [0], "code_ids": [1]},
        max_alpha=64.0,
        tol=0.02,
    )
    assert reached is False
    assert alpha == 64.0
    assert achieved < SAT + 1.0  # saturated below the goal
    assert achieved == pytest.approx(SAT, rel=1e-6)


def test_match_effect_zero_target_is_noop(monkeypatch):
    monkeypatch.setattr(cc, "target_te", _fake_te)
    assert match_effect(
        object(),
        "p",
        8,
        np.zeros(4),
        0.0,
        {"comment_ids": [0], "code_ids": [1]},
    ) == (0.0, 0.0, True)


# --------------------------------------------------------------------------- #
# integration (shared Granite daemon; skipped when it is down)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.skipif(not daemon_available(), reason="steering daemon not running")
def test_match_effect_reaches_small_target_via_daemon():
    client = DaemonClient()
    health = client.health()
    assert health["ok"] is True
    assert health["vocab"] == 49159

    token_sets = cc.build_token_sets(client, ["#", "# ", " ##"], ["return", "def", "for"])
    assert token_sets["comment_ids"] and token_sets["code_ids"]

    vec = client.build_vector(
        8,
        ["# add two numbers and return the result\n"],
        ["def add(a, b):\n"],
        chat=True,
        position="pooled",
    )["vec"]
    assert float(np.linalg.norm(vec)) > 1e-3

    prompt = "def merge(a, b):\n"
    probe = cc.target_te(client, prompt, 8, vec, 4.0, token_sets, chat=True)
    if abs(probe) < 0.02:
        pytest.skip(f"direction has no usable effect on this prompt (te={probe:.4f})")

    target = 0.5 * probe
    alpha, achieved, reached = match_effect(
        client,
        prompt,
        8,
        vec,
        target,
        token_sets,
        chat=True,
        max_alpha=4.0,
        tol=0.06,
    )
    assert reached is True, f"could not reach {target:.3f}; got {achieved:.3f} at {alpha:.2f}"
    assert achieved == pytest.approx(target, rel=0.06 + 1e-9)
    assert 0.0 < alpha < 4.0
