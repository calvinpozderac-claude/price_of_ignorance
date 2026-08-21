"""Idiosyncratic driver error: magnitude sigma, correlation rho."""

import os

import numpy as np
import pytest

from poi import solve_single_class, square_lattice
from poi.stochastic import (
    dial_logit,
    draw_errors,
    solve_stochastic,
    solve_stochastic_bpr,
)

BASE = "/workspace/bstabler/transportationnetworks"


# ------------------------------------------------------------ the error draw --
@pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_sigma_sets_magnitude_and_rho_sets_correlation(rho):
    """The two knobs must move independently: Var = sigma^2 for every rho."""
    eps = draw_errors(400, 3000, 0.3, rho, rng=0)
    assert eps.std() == pytest.approx(0.3, rel=0.03)
    corr = np.corrcoef(eps[0], eps[1])[0, 1]
    assert corr == pytest.approx(rho, abs=0.06)


def test_errors_skip_connector_links():
    net = square_lattice(6, 0.5, rng=0)
    eps = draw_errors(net.n_edges, 8, 0.5, 0.0, rng=1, mask=net.is_real_road)
    assert np.all(eps[:, ~net.is_real_road] == 0.0)
    assert np.any(eps[:, net.is_real_road] != 0.0)


def test_draw_errors_rejects_bad_input():
    with pytest.raises(ValueError):
        draw_errors(10, 2, 0.1, 1.5)
    with pytest.raises(ValueError):
        draw_errors(10, 2, -0.1, 0.5)


# ------------------------------------------------------- the multi-class QP --
def test_zero_sigma_reproduces_the_deterministic_endpoints():
    net = square_lattice(8, 0.6447, rng=3)
    _, eq, _ = solve_single_class(net, "equilibrium")
    _, opt, _ = solve_single_class(net, "optimum")
    assert solve_stochastic(net, 0.0, 0.0, 4, alpha=0.0).total_cost == pytest.approx(eq, rel=1e-6)
    assert solve_stochastic(net, 0.0, 0.0, 4, alpha=1.0).total_cost == pytest.approx(opt, rel=1e-6)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_flow_is_conserved(alpha):
    net = square_lattice(8, 0.6, rng=2)
    s = solve_stochastic(net, 0.3, 0.4, 8, alpha=alpha, rng=5)
    assert np.allclose(net.incidence @ s.x, net.demand(), atol=1e-7)
    assert s.x.min() > -1e-9


def test_no_population_beats_the_optimum():
    net = square_lattice(8, 0.6447, rng=4)
    _, opt, _ = solve_single_class(net, "optimum")
    for sig in (0.1, 0.4):
        for rho in (0.0, 1.0):
            s = solve_stochastic(net, sig, rho, 8, alpha=0.5, rng=6)
            assert s.total_cost >= opt - 1e-7


def test_solve_stochastic_rejects_bad_K():
    with pytest.raises(ValueError):
        solve_stochastic(square_lattice(4, 0.5, rng=0), 0.1, 0.0, 0)


# --------------------------------------------------- the headline comparison --
def _mean(nets, sigma, rho, K, seed0=100):
    return float(np.mean([
        solve_stochastic(n, sigma, rho, K, rng=seed0 + i).total_cost
        for i, n in enumerate(nets)
    ]))


def test_independent_error_helps_but_correlated_error_hurts():
    """Same magnitude, opposite sign -- correlation is the control variable."""
    nets = [square_lattice(8, 0.6447, rng=s) for s in range(6)]
    base = _mean(nets, 0.0, 0.0, 1)
    indep = _mean(nets, 0.15, 0.0, 48)
    corr = _mean(nets, 0.15, 1.0, 48)
    assert indep < base, "independent error should relieve congestion"
    assert corr > base, "correlated error should be harmful"
    assert corr > indep


def test_few_types_look_correlated():
    """K is itself a correlation knob: a handful of types is a handful of groups."""
    nets = [square_lattice(8, 0.6447, rng=s) for s in range(6)]
    base = _mean(nets, 0.0, 0.0, 1)
    coarse = _mean(nets, 0.10, 0.0, 4) / base
    fine = _mean(nets, 0.10, 0.0, 64) / base
    assert fine < coarse
    assert fine < 1.0


# ------------------------------------------------------------- logit SUE -----
def test_dial_conserves_flow_exactly():
    """MSA must start from a feasible loading, not from zero."""
    net = square_lattice(8, 0.6447, rng=0)
    for theta in (100.0, 10.0, 1.0):
        x, _, _ = dial_logit(net, theta, max_iter=1500, tol=1e-12)
        assert x[net.tail == net.source].sum() == pytest.approx(1.0, abs=1e-9)
        assert np.allclose(net.incidence @ x, net.demand(), atol=1e-8)


def test_dial_large_theta_approaches_the_user_equilibrium():
    net = square_lattice(8, 0.6447, rng=0)
    _, eq, _ = solve_single_class(net, "equilibrium")
    _, C, _ = dial_logit(net, 200.0, max_iter=3000, tol=1e-12)
    assert C == pytest.approx(eq, rel=2e-3)


def test_logit_dispersion_is_u_shaped():
    """An independent check on the K-type result, with Gumbel path errors."""
    net = square_lattice(8, 0.6447, rng=0)
    _, eq, _ = solve_single_class(net, "equilibrium")
    curve = [dial_logit(net, t, max_iter=2000, tol=1e-12)[1] / eq
             for t in (200.0, 10.0, 1.0)]
    assert curve[1] < curve[0]      # some dispersion helps
    assert curve[2] > 1.0           # too much is far worse than the UE
    assert curve[1] < 0.995


def test_dial_rejects_bad_input():
    net = square_lattice(4, 0.5, rng=0)
    with pytest.raises(ValueError):
        dial_logit(net, 0.0)


# ------------------------------------------------------------ BPR networks ---
@pytest.mark.skipif(not os.path.isdir(BASE), reason="benchmark data not cloned")
def test_bpr_zero_sigma_matches_the_two_class_solver():
    from poi.bpr import frank_wolfe_mixed, read_tntp

    net = read_tntp(os.path.join(BASE, "SiouxFalls"))
    for alpha in (0.0, 1.0):
        a = frank_wolfe_mixed(net, alpha, tol=1e-6, max_iter=600).total_cost
        b = solve_stochastic_bpr(net, 0.0, 0.0, 3, alpha, rng=0,
                                 tol=1e-6, max_iter=600).total_cost
        assert b == pytest.approx(a, rel=1e-4)


@pytest.mark.skipif(not os.path.isdir(BASE), reason="benchmark data not cloned")
def test_bpr_perceived_free_flow_stays_positive():
    """Large sigma must not produce negative perceived travel times."""
    from poi.bpr import read_tntp

    net = read_tntp(os.path.join(BASE, "SiouxFalls"))
    r = solve_stochastic_bpr(net, 2.0, 0.0, 4, 0.0, rng=1, tol=1e-5, max_iter=200)
    assert np.isfinite(r.total_cost)
    assert r.x.min() > -1e-9
