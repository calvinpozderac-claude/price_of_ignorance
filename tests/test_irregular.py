"""Which findings survive when the lattice's equal-path-length structure goes?

The lattice makes every source-to-sink path cross exactly ``2L`` roads, so
``sum_e x_e`` is the same for every feasible flow.  That is the "free direction"
the gamma-star derivation uses, and it also means ignorance can never send a
driver down a *longer* route.  ``skip_dag`` keeps everything else identical and
breaks only that, which is what makes these comparisons interpretable.
"""

import numpy as np
import pytest

from poi import solve_single_class, square_lattice
from poi.ignorance import OMEGA_STAR, solve_ignorant
from poi.irregular import path_length_spread, skip_dag

SEEDS = range(8)


def _mean(nets, omega, alpha):
    return float(np.mean([solve_ignorant(n, omega, alpha).total_cost for n in nets]))


# ------------------------------------------------------------ the structure --
def test_lattice_has_uniform_path_length():
    assert path_length_spread(square_lattice(10, 0.5, rng=0)) == (20, 20)


def test_skip_dag_breaks_uniform_path_length():
    assert path_length_spread(skip_dag(24, 8, 0.5, rng=0, skip_frac=0.0)) == (24, 24)
    lo, hi = path_length_spread(skip_dag(24, 8, 0.5, rng=0, skip_frac=0.6))
    assert lo < hi


def test_total_link_usage_is_pinned_on_the_lattice_only():
    """sum_e x_e is constant on the lattice and varies on the skip-DAG."""
    lat = square_lattice(10, 0.6, rng=1)
    tot = [solve_ignorant(lat, w, 0.0).x[lat.is_real_road].sum() for w in (0.0, 0.5, 0.9)]
    assert np.allclose(tot, 20.0, atol=1e-6)

    dag = skip_dag(24, 10, 0.6, rng=1, skip_frac=0.6)
    tot = [solve_ignorant(dag, w, 0.0).x[dag.is_real_road].sum() for w in (0.0, 0.5, 0.9)]
    assert max(tot) - min(tot) > 0.5


def test_skip_dag_rejects_bad_input():
    with pytest.raises(ValueError):
        skip_dag(10, 5, 0.5, skip_frac=1.5)
    with pytest.raises(ValueError):
        skip_dag(10, 5, 1.5)


# ------------------------------------------------------- what still survives --
@pytest.mark.parametrize("p", [0.35, 0.55])
def test_altruism_still_helps_with_accurate_knowledge(p):
    nets = [skip_dag(24, 10, p, rng=s, skip_frac=0.5) for s in SEEDS]
    assert _mean(nets, 0.0, 1.0) < _mean(nets, 0.0, 0.0)


@pytest.mark.parametrize("p", [0.35, 0.55])
def test_altruism_still_reverses_sign_under_ignorance(p):
    """The headline finding is not an artefact of equal path lengths."""
    nets = [skip_dag(24, 10, p, rng=s, skip_frac=0.5) for s in SEEDS]
    assert _mean(nets, 0.85, 1.0) > _mean(nets, 0.85, 0.0)


def test_no_population_beats_the_true_optimum():
    nets = [skip_dag(24, 10, 0.5, rng=s, skip_frac=0.5) for s in SEEDS]
    opt = float(np.mean([solve_single_class(n, "optimum")[1] for n in nets]))
    for w in (0.0, 0.4, 0.85):
        for a in (0.0, 0.5, 1.0):
            assert _mean(nets, w, a) >= opt - 1e-6


# --------------------------------------------------------- what does *not* --
def test_universal_benefit_of_ignorance_is_lattice_specific():
    """On the lattice, omega = 2/3 helps at every p (arXiv:2503.09684).

    With unequal path lengths that fails: ignorance can route traffic onto
    genuinely longer paths, and at high p -- where the congestible backbone is
    exactly what should be used -- it is strongly harmful.
    """
    lat = [square_lattice(12, 0.75, rng=s) for s in SEEDS]
    assert _mean(lat, OMEGA_STAR, 0.0) < _mean(lat, 0.0, 0.0)      # helps

    dag = [skip_dag(24, 10, 0.75, rng=s, skip_frac=0.5) for s in SEEDS]
    assert _mean(dag, OMEGA_STAR, 0.0) > _mean(dag, 0.0, 0.0)      # hurts


def test_ignorance_damage_grows_with_congestible_fraction():
    """The more the network relies on its fast backbone, the worse ignorance is."""
    pi = []
    for p in (0.25, 0.55, 0.85):
        nets = [skip_dag(24, 10, p, rng=s, skip_frac=0.5) for s in SEEDS]
        pi.append(_mean(nets, OMEGA_STAR, 0.0) / _mean(nets, 0.0, 0.0))
    assert pi[0] < pi[1] < pi[2]
    assert pi[-1] > 1.2
