"""User ignorance: reproduction of arXiv:2503.09684 and its interaction with altruism.

Notation: ``omega`` is the ignorance level (that paper's ``alpha``); ``alpha``
remains the altruistic fraction from the price-of-anarchy work.
"""

import numpy as np
import pytest

from poi import pigou_network, solve_mixed, solve_single_class, square_lattice
from poi.ignorance import (
    OMEGA_STAR,
    objective_identity_residual,
    perceive,
    solve_ignorant,
)
from poi.pigou import solve_pigou, solve_pigou_ignorant
from poi.qp import solve_uniform

OMEGAS = [0.0, 0.2, 0.4, 2 / 3, 0.8, 1.0]
QP_TOL = 1e-5


# ----------------------------------------------------------- the cost mixing
def test_perceive_endpoints():
    """omega = 0 leaves costs alone; omega = 1 makes every road look identical."""
    net = square_lattice(8, 0.6, rng=0)
    same = perceive(net, 0.0)
    assert np.allclose(same.a, net.a) and np.allclose(same.b, net.b)

    blind = perceive(net, 1.0)
    real = net.is_real_road
    assert np.allclose(blind.a[real], 0.5)
    assert np.allclose(blind.b[real], 0.5)


@pytest.mark.parametrize("omega", OMEGAS)
def test_perceive_is_the_papers_weighted_average(omega):
    """c^p = (1 - omega/2) * true + (omega/2) * opposite-type cost."""
    net = square_lattice(6, 0.5, rng=1)
    q = perceive(net, omega)
    u = omega / 2
    fast, slow = net.b > 0, (net.b == 0) & (net.a > 0)
    assert np.allclose(q.a[slow], 1 - u) and np.allclose(q.b[slow], u)
    assert np.allclose(q.a[fast], u) and np.allclose(q.b[fast], 1 - u)
    # Connector edges carry no cost and nothing to be ignorant about.
    assert np.allclose(q.a[~net.is_real_road], 0.0)


def test_perceive_rejects_out_of_range():
    net = square_lattice(4, 0.5, rng=0)
    with pytest.raises(ValueError):
        perceive(net, 1.5)


# ------------------------------------------------- reproduction of the paper
@pytest.mark.parametrize("omega", OMEGAS)
def test_pigou_with_ignorance(omega):
    """Paper: currents (omega/2, 1 - omega/2) and PI = 1 - (omega/2)(1 - omega/2)."""
    exact = solve_pigou_ignorant(0.0, omega)
    assert exact.x_slow == pytest.approx(omega / 2)
    assert exact.x_fast == pytest.approx(1 - omega / 2)
    PI = exact.total_cost / solve_pigou_ignorant(0.0, 0.0).total_cost
    assert PI == pytest.approx(1 - (omega / 2) * (1 - omega / 2))
    assert PI <= 1 + 1e-12  # any ignorance helps, in Pigou


@pytest.mark.parametrize("omega", OMEGAS)
@pytest.mark.parametrize("alpha", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_pigou_closed_form_matches_qp(omega, alpha):
    net = pigou_network()
    exact = solve_pigou_ignorant(alpha, omega)
    qp = solve_ignorant(net, omega, alpha)
    assert qp.status == "Solved"
    assert qp.total_cost == pytest.approx(exact.total_cost, abs=QP_TOL)
    assert qp.x[1] == pytest.approx(exact.x_fast, abs=QP_TOL)


@pytest.mark.parametrize("omega", OMEGAS)
def test_pigou_altruists_always_target_the_true_optimum(omega):
    """Special to Pigou: the perceived optimum coincides with the true one."""
    assert solve_pigou_ignorant(1.0, omega).x_fast == pytest.approx(0.5)


@pytest.mark.parametrize("L,p", [(8, 0.3), (8, 0.6447), (10, 0.8)])
def test_total_ignorance_gives_uniform_flow(L, p):
    """At omega = 1 all roads look alike, so x = 1/(2L) and C = p + 2L(1 - p)."""
    net = square_lattice(L, p, rng=4)
    s = solve_ignorant(net, 1.0, 0.0)
    real = net.is_real_road
    assert np.allclose(s.x[real], 1 / (2 * L), atol=1e-8)
    frac = net.meta["congestible_mask"][: net.meta["n_lattice_edges"]].mean()
    assert s.total_cost == pytest.approx(frac + 2 * L * (1 - frac), rel=1e-7)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_total_ignorance_makes_altruism_irrelevant(alpha):
    """If every road looks identical, altruists and selfish behave identically."""
    net = square_lattice(8, 0.6447, rng=2)
    base = solve_ignorant(net, 1.0, 0.0).total_cost
    assert solve_ignorant(net, 1.0, alpha).total_cost == pytest.approx(base, rel=1e-7)


def test_objective_identity_at_two_thirds():
    """Paper Eq. 12: F_{2/3} = C/3 + (1/6) sum_slow x^2 + const, for ANY flow."""
    net = square_lattice(10, 0.6447, rng=3)
    for alpha in (0.0, 0.5, 1.0):
        x = solve_ignorant(net, OMEGA_STAR, alpha).x
        assert objective_identity_residual(net, x) < 1e-10
    # It is an algebraic identity, so it must hold for a non-equilibrium flow too.
    x = solve_ignorant(net, 0.3, 0.7).x
    assert objective_identity_residual(net, x) < 1e-10


@pytest.mark.parametrize("p", [0.15, 0.35, 0.55, 0.6447, 0.75, 0.9])
@pytest.mark.parametrize("omega", [0.2, 0.45, 2 / 3])
def test_ignorance_below_two_thirds_always_helps(p, omega):
    """The paper's central claim: PI <= 1 for every p when omega <= 2/3."""
    tot_ign = tot_kno = 0.0
    for seed in range(6):
        net = square_lattice(12, p, rng=seed)
        tot_ign += solve_ignorant(net, omega, 0.0).total_cost
        tot_kno += solve_ignorant(net, 0.0, 0.0).total_cost
    assert tot_ign / tot_kno <= 1.0 + 1e-9


def test_ignorance_is_most_useful_near_percolation():
    """Minimum price of ignorance ~0.95, located near p_c at omega ~ 2/3."""
    ps = np.linspace(0.2, 0.95, 16)
    PI = []
    for p in ps:
        ign = kno = 0.0
        for seed in range(8):
            net = square_lattice(20, float(p), rng=seed)
            ign += solve_ignorant(net, OMEGA_STAR, 0.0).total_cost
            kno += solve_ignorant(net, 0.0, 0.0).total_cost
        PI.append(ign / kno)
    PI = np.array(PI)
    assert PI.min() == pytest.approx(0.95, abs=0.02)
    assert abs(ps[PI.argmin()] - 0.6447) < 0.16


# ------------------------------------------- ignorance crossed with altruism
@pytest.mark.parametrize("omega", [0.0, 0.3, 2 / 3, 0.9])
def test_no_population_beats_the_true_optimum(omega):
    """Whatever drivers believe, they cannot do better than the true optimum."""
    for seed in range(3):
        net = square_lattice(10, 0.6447, rng=seed)
        _, C_opt, _ = solve_single_class(net, "optimum")
        for alpha in (0.0, 0.4, 1.0):
            assert solve_ignorant(net, omega, alpha).total_cost >= C_opt - 1e-7


def test_altruism_hurts_at_high_ignorance():
    """The headline finding: past omega ~ 2/3 adding altruists raises the cost."""
    for p in (0.45, 0.6447, 0.8):
        lo = hi = 0.0
        for seed in range(8):
            net = square_lattice(14, p, rng=seed)
            lo += solve_ignorant(net, OMEGA_STAR, 0.0).total_cost
            hi += solve_ignorant(net, OMEGA_STAR, 1.0).total_cost
        assert hi > lo, f"expected altruism to hurt at p={p}, omega=2/3"


def test_altruism_helps_at_low_ignorance():
    """...while with accurate knowledge it still helps, as in the first study."""
    for p in (0.45, 0.6447):
        lo = hi = 0.0
        for seed in range(8):
            net = square_lattice(14, p, rng=seed)
            lo += solve_ignorant(net, 0.0, 0.0).total_cost
            hi += solve_ignorant(net, 0.0, 1.0).total_cost
        assert hi < lo


# ----------------------------------- the analytic altruism/ignorance trade-off
def gamma_star(omega):
    """Uniform altruism level that offsets ignorance ``omega``: (2-3w)/(2-w)."""
    return (2 - 3 * omega) / (2 - omega)


def test_gamma_star_endpoints():
    assert gamma_star(0.0) == pytest.approx(1.0)
    assert gamma_star(OMEGA_STAR) == pytest.approx(0.0)
    assert gamma_star(0.9) < 0  # past 2/3 you would need spite, not altruism


@pytest.mark.parametrize("omega", [0.0, 0.25, 0.5, 2 / 3, 0.8])
@pytest.mark.parametrize("p", [0.45, 0.6447])
def test_substitution_curve(p, omega):
    """A uniform population at gamma*(omega) reproduces the social optimum."""
    net = square_lattice(16, p, rng=11)
    _, C_opt, _ = solve_single_class(net, "optimum")
    q = perceive(net, omega)
    C = net.total_cost(solve_uniform(q, gamma_star(omega)))
    # Within 0.5% of the true optimum: the residual sum_slow x^2 term that the
    # paper argues is subleading at large L is what stops it being exact.
    assert C == pytest.approx(C_opt, rel=5e-3)
    assert C >= C_opt - 1e-7


@pytest.mark.parametrize("omega", [0.0, 0.3, 0.6])
def test_gamma_star_is_a_local_minimum(omega):
    """Moving off gamma* in either direction costs more."""
    net = square_lattice(16, 0.6447, rng=5)
    q = perceive(net, omega)
    g = gamma_star(omega)
    at = net.total_cost(solve_uniform(q, g))
    for d in (-0.3, +0.3):
        assert net.total_cost(solve_uniform(q, g + d)) >= at - 1e-9


def test_solve_uniform_matches_the_two_known_cases():
    """gamma = 0 is the selfish equilibrium; gamma = 1 is the social optimum."""
    net = square_lattice(12, 0.6447, rng=7)
    _, C_opt, _ = solve_single_class(net, "optimum")
    _, C_eq, _ = solve_single_class(net, "equilibrium")
    assert net.total_cost(solve_uniform(net, 0.0)) == pytest.approx(C_eq, rel=1e-7)
    assert net.total_cost(solve_uniform(net, 1.0)) == pytest.approx(C_opt, rel=1e-7)
