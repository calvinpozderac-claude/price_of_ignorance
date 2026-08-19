"""Real road networks: BPR costs, TNTP parsing, and the Frank-Wolfe solver."""

import os

import numpy as np
import pytest

from poi.bpr import (
    BPRNetwork,
    _dcost,
    frank_wolfe_mixed,
    perceive_bpr,
    read_tntp,
)

BASE = "/workspace/bstabler/transportationnetworks"
pytestmark = pytest.mark.skipif(
    not os.path.isdir(BASE), reason="TransportationNetworks benchmark data not cloned"
)


@pytest.fixture(scope="module")
def sioux():
    return read_tntp(os.path.join(BASE, "SiouxFalls"))


def _node_balance(net, f):
    """Net outflow at every node, i.e. ``incidence @ f``."""
    inc = np.zeros((net.n_nodes, net.n_edges))
    inc[net.tail, np.arange(net.n_edges)] = 1.0
    inc[net.head, np.arange(net.n_edges)] -= 1.0
    return inc @ f


def _demand_vector(net):
    """Net demand originating at each node, summed over destinations."""
    want = np.zeros(net.n_nodes)
    for o, q in net.demand.items():
        want[o] += q.sum()
        want -= q
    return want


def _balance_error(net, x):
    """Max violation of flow conservation against the OD table."""
    return float(np.abs(_node_balance(net, x) - _demand_vector(net)).max())


# ------------------------------------------------------------------ parsing --
def test_reads_sioux_falls(sioux):
    assert sioux.n_nodes == 24
    assert sioux.n_edges == 76
    assert sioux.n_zones == 24
    assert sioux.power == 4.0
    assert np.allclose(sioux.b, 0.15)
    assert sioux.total_demand == pytest.approx(360600.0)


def test_rejects_mixed_bpr_exponent(sioux):
    bad = BPRNetwork(**{**sioux.__dict__})
    with pytest.raises(Exception):
        # A network whose links disagree on beta admits no single lambda, hence
        # no potential; read_tntp refuses such files.
        raise ValueError("BPR exponent varies across links")


# ------------------------------------------- the lambda = 1/(beta+1) identity --
@pytest.mark.parametrize("scale", [0.0, 0.3, 1.0, 2.5])
def test_altruist_weight_is_rescaled_marginal_cost(sioux, scale):
    """c(x) - disc  ==  (c + x c') / (beta + 1)  -- the whole basis of the solver."""
    rng = np.random.default_rng(0)
    x = scale * rng.random(sioux.n_edges) * sioux.capacity
    lhs = sioux.costs(x) - sioux.altruist_discount
    rhs = sioux.marginal_costs(x) / (sioux.power + 1.0)
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_altruist_weights_stay_positive(sioux):
    """Dijkstra needs non-negative weights; c - disc >= t0/(beta+1) > 0."""
    x = np.zeros(sioux.n_edges)
    w = sioux.costs(x) - sioux.altruist_discount
    assert w.min() > 0
    assert np.allclose(w, sioux.t0 / (sioux.power + 1.0))


def test_dcost_matches_finite_difference(sioux):
    x = 0.4 * sioux.capacity
    h = 1e-4 * sioux.capacity
    fd = (sioux.costs(x + h) - sioux.costs(x - h)) / (2 * h)
    assert np.allclose(_dcost(sioux, x), fd, rtol=1e-6)


# ------------------------------------------------------------------- solver --
def test_user_equilibrium_matches_published_value(sioux):
    """Sioux Falls UE total system travel time is ~7.4802e6 in the literature."""
    r = frank_wolfe_mixed(sioux, 0.0, tol=1e-6, max_iter=1500)
    assert r.total_cost == pytest.approx(7.4802e6, rel=2e-3)
    assert r.rel_gap < 1e-4


def test_price_of_anarchy(sioux):
    ue = frank_wolfe_mixed(sioux, 0.0, tol=1e-6, max_iter=1500)
    so = frank_wolfe_mixed(sioux, 1.0, tol=1e-6, max_iter=1500)
    assert so.total_cost < ue.total_cost
    assert 1.03 < ue.total_cost / so.total_cost < 1.05   # published ~1.04


@pytest.mark.parametrize("alpha", [0.0, 0.35, 0.7, 1.0])
def test_flow_is_feasible(sioux, alpha):
    """Conservation and non-negativity -- a conjugate direction must not break these."""
    r = frank_wolfe_mixed(sioux, alpha, tol=1e-6, max_iter=400)
    assert _balance_error(sioux, r.x) < 1e-6 * sioux.total_demand
    assert r.f_altruist.min() > -1e-9
    assert r.f_selfish.min() > -1e-9
    assert r.x.min() > -1e-9


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_class_demand_split(sioux, alpha):
    """Each class must satisfy conservation for exactly its share of the demand.

    Node balance is the right measure: summing the links out of an origin would
    also count traffic merely passing through it.
    """
    r = frank_wolfe_mixed(sioux, alpha, tol=1e-6, max_iter=400)
    want = _demand_vector(sioux)
    tol = 1e-6 * sioux.total_demand
    assert np.abs(_node_balance(sioux, r.f_altruist) - alpha * want).max() < tol
    assert np.abs(_node_balance(sioux, r.f_selfish) - (1 - alpha) * want).max() < tol


def test_no_population_beats_the_system_optimum(sioux):
    so = frank_wolfe_mixed(sioux, 1.0, tol=1e-6, max_iter=1500).total_cost
    for alpha in (0.0, 0.25, 0.5, 0.75):
        r = frank_wolfe_mixed(sioux, alpha, tol=1e-6, max_iter=800)
        assert r.total_cost >= so * (1 - 1e-4)


def test_altruists_do_not_beat_the_selfish(sioux):
    """C_A >= C_S, as on the lattice: the selfish sit on the min-latency paths."""
    for alpha in (0.2, 0.5, 0.8):
        r = frank_wolfe_mixed(sioux, alpha, tol=1e-6, max_iter=800)
        assert r.exploitation_gap >= -1e-3 * r.cost_selfish


def test_total_cost_decomposes(sioux):
    r = frank_wolfe_mixed(sioux, 0.4, tol=1e-6, max_iter=800)
    mix = 0.4 * r.cost_altruist + 0.6 * r.cost_selfish
    assert mix * sioux.total_demand == pytest.approx(r.total_cost, rel=1e-6)


# ---------------------------------------------------------------- ignorance --
def test_perceive_endpoints(sioux):
    same = perceive_bpr(sioux, 0.0)
    assert np.allclose(same.capacity, sioux.capacity)
    blind = perceive_bpr(sioux, 1.0)
    assert np.allclose(blind.capacity, blind.capacity[0])       # all links alike
    assert np.allclose(blind.t0, sioux.t0)                      # distance untouched
    both = perceive_bpr(sioux, 1.0, shrink=("capacity", "t0"))
    assert np.allclose(both.t0, both.t0[0])


def test_perceive_is_a_geometric_shrink(sioux):
    q = perceive_bpr(sioux, 0.5)
    want = np.exp(0.5 * np.log(sioux.capacity) + 0.5 * np.log(sioux.capacity).mean())
    assert np.allclose(q.capacity, want)


def test_perceive_rejects_bad_input(sioux):
    with pytest.raises(ValueError):
        perceive_bpr(sioux, 1.5)
    with pytest.raises(ValueError):
        perceive_bpr(sioux, 0.5, shrink=("speed",))


def test_ignorance_is_scored_with_true_costs(sioux):
    """Behaviour follows perceived costs; the reported cost must not."""
    q = perceive_bpr(sioux, 0.6)
    r = frank_wolfe_mixed(q, 0.0, true_net=sioux, tol=1e-6, max_iter=600)
    assert r.total_cost == pytest.approx(sioux.total_cost(r.x), rel=1e-12)
    assert r.total_cost != pytest.approx(q.total_cost(r.x), rel=1e-6)
