"""Structural checks that the lattice matches the one described in the paper."""

import numpy as np
import pytest

from poi import square_lattice, solve_mixed, solve_single_class, uniform_entry_demand


@pytest.mark.parametrize("L", [1, 2, 5, 10])
def test_counts_match_paper(L):
    """Paper quotes 4L^2 roads, L boundary vertices and path length 2L."""
    net = square_lattice(L, 0.5, rng=0)
    assert net.meta["n_lattice_edges"] == 4 * L**2
    # 4L^2 lattice roads + L source connectors + L sink connectors
    assert net.n_edges == 4 * L**2 + 2 * L
    assert net.meta["path_length"] == 2 * L
    # (2L + 1) layers of L vertices, plus super-source and super-sink
    assert net.n_nodes == (2 * L + 1) * L + 2


@pytest.mark.parametrize("L", [3, 6, 10])
def test_number_of_paths(L):
    """Each left vertex must reach the far side by 2**(2L) distinct paths."""
    net = square_lattice(L, 0.0, rng=0)
    counts = np.zeros(net.n_nodes)
    counts[net.source] = 1.0
    # Nodes are numbered layer-major, so the source (highest id) is relayed first
    # and the remaining edges are already in topological order.
    src_edges = np.flatnonzero(net.tail == net.source)
    for e in src_edges:
        counts[net.head[e]] += counts[net.tail[e]]
    for e in np.argsort(net.tail, kind="stable"):
        if net.tail[e] == net.source:
            continue
        counts[net.head[e]] += counts[net.tail[e]]
    # L entry vertices, each reaching 2**(2L) paths
    assert counts[net.sink] == pytest.approx(L * 2.0 ** (2 * L))


@pytest.mark.parametrize("L", [4, 8])
def test_p_zero_all_constant_roads(L):
    """Every road costs 1 and every path is 2L long, so C = 2L exactly."""
    net = square_lattice(L, 0.0, rng=3)
    for alpha in (0.0, 0.5, 1.0):
        s = solve_mixed(net, alpha)
        assert s.total_cost == pytest.approx(2 * L, abs=1e-6)


@pytest.mark.parametrize("L", [4, 8])
def test_p_one_symmetry(L):
    """At p = 1 symmetry forces x = 1/(2L) on all 4L^2 roads, so C = 1."""
    net = square_lattice(L, 1.0, rng=3)
    for alpha in (0.0, 0.5, 1.0):
        s = solve_mixed(net, alpha)
        assert s.total_cost == pytest.approx(1.0, abs=1e-6)
    road = net.meta["congestible_mask"]
    s = solve_mixed(net, 0.0)
    assert np.allclose(s.x[road], 1.0 / (2 * L), atol=1e-6)


def test_congestible_fraction_matches_p():
    net = square_lattice(40, 0.6, rng=12)
    frac = net.meta["congestible_mask"][: net.meta["n_lattice_edges"]].mean()
    assert frac == pytest.approx(0.6, abs=0.02)


def test_uniform_entry_gives_same_qualitative_answer():
    """The busbar and forced-uniform boundary conditions must agree closely."""
    for seed in range(4):
        a = square_lattice(8, 0.6447, rng=seed, entry="busbar")
        b = square_lattice(8, 0.6447, rng=seed, entry="uniform")
        ca = solve_mixed(a, 0.0).total_cost
        cb = solve_mixed(b, 0.0, demand=uniform_entry_demand(b)).total_cost
        # Forcing uniform entry only removes freedom, so it cannot be cheaper.
        assert cb >= ca - 1e-7
        assert cb == pytest.approx(ca, rel=0.15)


def test_incidence_conserves_flow():
    net = square_lattice(6, 0.5, rng=1)
    s = solve_mixed(net, 0.4)
    assert np.allclose(net.incidence @ s.x, net.demand(), atol=1e-8)
    assert np.allclose(net.incidence @ s.f_altruist, net.demand(0.4), atol=1e-8)
    assert np.allclose(net.incidence @ s.f_selfish, net.demand(0.6), atol=1e-8)
