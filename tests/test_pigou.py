"""The exact two-road solution, and its agreement with the QP solver."""

import numpy as np
import pytest

from poi import pigou_curves, pigou_network, solve_mixed, solve_pigou

ALPHAS = np.linspace(0.0, 1.0, 41)


def test_paper_endpoints():
    """Pigou's own numbers: C_eq = 1, C_opt = 3/4, POA = 4/3."""
    assert solve_pigou(0.0).total_cost == pytest.approx(1.0)
    assert solve_pigou(1.0).total_cost == pytest.approx(0.75)
    assert solve_pigou(0.0).total_cost / solve_pigou(1.0).total_cost == pytest.approx(4 / 3)
    assert solve_pigou(1.0).x_fast == pytest.approx(0.5)


# Interior-point methods approach vertex solutions from strictly inside the
# feasible set, so a solution pinned to a corner (e.g. alpha = 0, where all
# traffic sits on road 2) carries a small residual offset. 1e-5 is still three
# orders of magnitude below every effect measured in this project.
QP_TOL = 1e-5


@pytest.mark.parametrize("alpha", ALPHAS)
def test_closed_form_matches_qp(alpha):
    net = pigou_network()
    exact = solve_pigou(alpha)
    qp = solve_mixed(net, alpha)
    assert qp.status == "Solved"
    assert qp.total_cost == pytest.approx(exact.total_cost, abs=QP_TOL)
    assert qp.x[1] == pytest.approx(exact.x_fast, abs=QP_TOL)
    if 0 < alpha < 1:
        assert qp.cost_altruist == pytest.approx(exact.cost_altruist, abs=QP_TOL)
        assert qp.cost_selfish == pytest.approx(exact.cost_selfish, abs=QP_TOL)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_closed_form_cost_formula(alpha):
    """Below alpha* = 1/2 the algebra gives C = 1 - alpha + alpha^2, then C = 3/4."""
    got = solve_pigou(alpha).total_cost
    want = 1 - alpha + alpha**2 if alpha <= 0.5 else 0.75
    assert got == pytest.approx(want)


def test_saturation_at_one_half():
    """Half the population being altruistic already captures the whole gain."""
    cur = pigou_curves(ALPHAS)
    assert cur["alpha_star"] == pytest.approx(0.5)
    assert cur["efficiency"][ALPHAS == 0.5][0] == pytest.approx(1.0)
    # ...and below that the captured share is exactly 4 alpha (1 - alpha)
    lo = ALPHAS <= 0.5
    assert np.allclose(cur["efficiency"][lo], 4 * ALPHAS[lo] * (1 - ALPHAS[lo]))


def test_altruists_pay_and_selfish_free_ride():
    """Altruists sit at cost 1 while the selfish improve from 1 to 1/2."""
    for alpha in (0.1, 0.25, 0.4, 0.5):
        s = solve_pigou(alpha)
        assert s.cost_altruist == pytest.approx(1.0)
        assert s.cost_selfish == pytest.approx(1 - alpha)
        # Altruists do no better than under full anarchy; the selfish do better.
        assert s.cost_altruist >= solve_pigou(0.0).total_cost - 1e-12
        assert s.cost_selfish < solve_pigou(0.0).total_cost


def test_gap_peaks_at_saturation():
    """The altruist/selfish gap is largest exactly where the gains run out."""
    cur = pigou_curves(ALPHAS)
    interior = np.isfinite(cur["gap"])
    assert ALPHAS[interior][np.nanargmax(cur["gap"][interior])] == pytest.approx(0.5)
    assert np.nanmax(cur["gap"]) == pytest.approx(0.5)


def test_gap_above_saturation():
    """For alpha >= 1/2 the closed form gives C_A = 1/2 + 1/(4 alpha), C_S = 1/2."""
    for alpha in (0.5, 0.6, 0.75, 0.9, 1.0):
        s = solve_pigou(alpha)
        assert s.cost_altruist == pytest.approx(0.5 + 1 / (4 * alpha))
        if alpha < 1:
            assert s.cost_selfish == pytest.approx(0.5)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_altruists_never_beat_the_selfish(alpha):
    """Selfish drivers occupy the min-latency path, so C_A >= C_S always."""
    s = solve_pigou(alpha)
    if 0 < alpha < 1:
        assert s.exploitation_gap >= -1e-12


@pytest.mark.parametrize("k,a,b", [(1.0, 0.0, 1.0), (1.0, 0.2, 0.9), (2.0, 0.5, 3.0), (1.0, 0.0, 0.4)])
def test_general_parameters_match_qp(k, a, b):
    net = pigou_network(a2=a, b2=b, c1=k)
    for alpha in np.linspace(0, 1, 21):
        exact = solve_pigou(alpha, k=k, a=a, b=b)
        qp = solve_mixed(net, alpha)
        assert qp.total_cost == pytest.approx(exact.total_cost, abs=QP_TOL)


def test_regimes():
    """All three regimes of the closed form are reachable and self-consistent."""
    # Canonical case: at alpha = 0 the altruists (none) have retreated.
    assert solve_pigou(0.0, k=1.0, a=0.0, b=1.0).regime == "retreat"
    assert solve_pigou(0.9, k=1.0, a=0.0, b=1.0).regime == "optimal"

    # b = 0.4 makes x_opt = 1: the fast road is never worth leaving, so the
    # selfish equilibrium is already optimal and alpha* = 0.
    flat = solve_pigou(0.0, k=1.0, a=0.0, b=0.4)
    assert flat.regime == "optimal"
    assert flat.x_fast == pytest.approx(1.0)

    # b = 4 makes x_eq = 0.25 < 1: road 2 saturates on selfish drivers alone,
    # the rest spill onto road 1, and both roads then cost exactly k.
    steep = solve_pigou(0.0, k=1.0, a=0.0, b=4.0)
    assert steep.regime == "saturated"
    assert steep.x_fast == pytest.approx(0.25)
    assert steep.cost_selfish == pytest.approx(1.0)
