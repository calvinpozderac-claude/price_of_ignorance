"""Correctness of the mixed-equilibrium QP itself."""

import numpy as np
import pytest

from poi import cost_range, solve_mixed, solve_single_class, square_lattice
from poi.qp import DEFAULT_RIDGE

SEEDS = range(4)
PS = [0.3, 0.5, 0.6447, 0.8]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("p", PS)
def test_endpoints_recover_the_papers_two_problems(p, seed):
    """alpha = 0 must be the selfish equilibrium and alpha = 1 the social optimum."""
    net = square_lattice(10, p, rng=seed)
    _, C_opt, st1 = solve_single_class(net, "optimum")
    _, C_eq, st2 = solve_single_class(net, "equilibrium")
    assert st1 == st2 == "Solved"
    assert solve_mixed(net, 0.0).total_cost == pytest.approx(C_eq, rel=1e-7)
    assert solve_mixed(net, 1.0).total_cost == pytest.approx(C_opt, rel=1e-7)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("p", PS)
def test_poa_is_between_one_and_four_thirds(p, seed):
    """Roughgarden-Tardos: for affine costs the POA cannot exceed 4/3."""
    net = square_lattice(10, p, rng=seed)
    poa = solve_mixed(net, 0.0).total_cost / solve_mixed(net, 1.0).total_cost
    assert 1.0 - 1e-9 <= poa <= 4 / 3 + 1e-9


@pytest.mark.parametrize("alpha", [0.0, 0.15, 0.4, 0.6, 0.85, 1.0])
def test_wardrop_conditions_hold(alpha):
    """Each class's used paths must all sit at that class's minimum perceived cost."""
    net = square_lattice(12, 0.6447, rng=5)
    s = solve_mixed(net, alpha)
    assert s.status == "Solved"
    assert s.residual_selfish < 1e-7
    assert s.residual_altruist < 1e-7
    # No used path may be cheaper than the class minimum either.
    if 0 < alpha < 1:
        assert s.cost_selfish >= s.mu_selfish - 1e-8


@pytest.mark.parametrize("alpha", [0.05, 0.2, 0.5, 0.7, 0.95])
@pytest.mark.parametrize("p", PS)
def test_altruists_never_beat_the_selfish(p, alpha):
    """C_A >= C_S: selfish drivers already occupy the minimum-latency paths."""
    for seed in SEEDS:
        net = square_lattice(10, p, rng=seed)
        s = solve_mixed(net, alpha)
        assert s.exploitation_gap >= -1e-8


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.5, 0.8, 1.0])
def test_flows_are_feasible(alpha):
    net = square_lattice(10, 0.6447, rng=2)
    s = solve_mixed(net, alpha)
    assert s.f_altruist.min() >= -1e-9
    assert s.f_selfish.min() >= -1e-9
    assert np.allclose(net.incidence @ s.f_altruist, net.demand(alpha), atol=1e-8)
    assert np.allclose(net.incidence @ s.f_selfish, net.demand(1 - alpha), atol=1e-8)


def test_total_cost_decomposes_across_classes():
    """C = alpha*C_A + (1-alpha)*C_S must hold identically."""
    net = square_lattice(10, 0.6447, rng=4)
    for alpha in (0.1, 0.35, 0.6, 0.9):
        s = solve_mixed(net, alpha)
        assert alpha * s.cost_altruist + (1 - alpha) * s.cost_selfish == pytest.approx(
            s.total_cost, rel=1e-8
        )


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_reported_cost_is_essentially_unique(alpha):
    """The tie-breaking ridge must not be doing any real work."""
    net = square_lattice(10, 0.6447, rng=0)
    s = solve_mixed(net, alpha)
    lo, hi = cost_range(net, s)
    assert hi - lo < 1e-5
    assert lo - 1e-6 <= s.total_cost <= hi + 1e-6


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.7, 1.0])
def test_ridge_independence(alpha):
    """Shrinking the ridge by three orders of magnitude changes nothing visible."""
    net = square_lattice(10, 0.6447, rng=1)
    a = solve_mixed(net, alpha, ridge=DEFAULT_RIDGE).total_cost
    b = solve_mixed(net, alpha, ridge=DEFAULT_RIDGE * 1e-3).total_cost
    assert a == pytest.approx(b, abs=1e-6)


def test_optimum_is_the_cheapest_achievable():
    """No mixed population can beat the true social optimum's total cost."""
    for seed in SEEDS:
        net = square_lattice(10, 0.6447, rng=seed)
        _, C_opt, _ = solve_single_class(net, "optimum")
        for alpha in np.linspace(0, 1, 11):
            assert solve_mixed(net, alpha).total_cost >= C_opt - 1e-7


def test_invalid_inputs():
    net = square_lattice(4, 0.5, rng=0)
    with pytest.raises(ValueError):
        solve_mixed(net, 1.5)
    with pytest.raises(ValueError):
        solve_single_class(net, "nonsense")
