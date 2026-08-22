"""Imitation dynamics on the altruistic fraction."""

import numpy as np
import pytest

from poi.evolution import (
    evolve,
    invasion_signs,
    rest_points,
    settle,
)
from poi.lattice import pigou_network
from poi.pigou import pigou_curves


ALPHAS = np.linspace(0.0, 1.0, 21)


def _endpoints_nan(D):
    """C_A is undefined at alpha=0 and C_S at alpha=1, as the solvers report."""
    D = np.asarray(D, dtype=float).copy()
    D[0] = D[-1] = np.nan
    return D


class TestRestPoints:
    def test_altruism_always_costly_collapses_to_selfish(self):
        D = _endpoints_nan(np.full(ALPHAS.size, 0.4))
        pts = rest_points(ALPHAS, D)
        assert [(p.alpha, p.stable) for p in pts] == [(0.0, True), (1.0, False)]

    def test_altruism_always_pays_fixates(self):
        D = _endpoints_nan(np.full(ALPHAS.size, -0.4))
        pts = rest_points(ALPHAS, D)
        assert [(p.alpha, p.stable) for p in pts] == [(0.0, False), (1.0, True)]

    def test_interior_attractor_is_found_and_marked_stable(self):
        # D crosses zero upwards at alpha = 0.4: altruists pay above it and are
        # paid below it, so the mixture is attracting.
        D = _endpoints_nan(ALPHAS - 0.4)
        pts = rest_points(ALPHAS, D)
        mixed = [p for p in pts if p.kind == "mixed"]
        assert len(mixed) == 1
        assert mixed[0].alpha == pytest.approx(0.4, abs=1e-3)
        assert mixed[0].stable
        assert not any(p.stable for p in pts if p.kind != "mixed")

    def test_interior_repeller_splits_the_basins(self):
        D = _endpoints_nan(0.4 - ALPHAS)
        pts = rest_points(ALPHAS, D)
        mixed = [p for p in pts if p.kind == "mixed"]
        assert len(mixed) == 1
        assert mixed[0].alpha == pytest.approx(0.4, abs=1e-3)
        assert not mixed[0].stable
        assert all(p.stable for p in pts if p.kind != "mixed")

    def test_mutation_pulls_the_endpoints_inward(self):
        D = _endpoints_nan(np.full(ALPHAS.size, 0.4))
        pts = rest_points(ALPHAS, D, mu=0.02)
        assert all(0.0 < p.alpha < 1.0 for p in pts)
        stable = [p for p in pts if p.stable]
        assert len(stable) == 1
        assert 0.0 < stable[0].alpha < 0.2


class TestInvasion:
    def test_signs_extrapolate_past_the_nan_endpoints(self):
        D = _endpoints_nan(ALPHAS - 0.5)
        d0, d1 = invasion_signs(ALPHAS, D)
        assert d0 == pytest.approx(-0.5, abs=1e-9)
        assert d1 == pytest.approx(0.5, abs=1e-9)

    def test_needs_two_finite_samples(self):
        with pytest.raises(ValueError):
            invasion_signs(ALPHAS, np.full(ALPHAS.size, np.nan))


class TestEvolve:
    @pytest.mark.parametrize("a0", [0.05, 0.5, 0.95])
    def test_costly_altruism_dies_from_any_start(self, a0):
        D = _endpoints_nan(np.full(ALPHAS.size, 0.4))
        assert evolve(ALPHAS, D, alpha0=a0)[-1] == pytest.approx(0.0, abs=1e-6)

    def test_basins_of_a_repeller(self):
        # D = 0.4 - alpha: altruists are paid above alpha = 0.4 and pay below
        # it, so 0.4 repels and the two endpoints share the interval.
        D = _endpoints_nan(0.4 - ALPHAS)
        assert evolve(ALPHAS, D, alpha0=0.3)[-1] == pytest.approx(0.0, abs=1e-6)
        assert evolve(ALPHAS, D, alpha0=0.5)[-1] == pytest.approx(1.0, abs=1e-6)

    def test_path_is_monotone_when_the_sign_never_changes(self):
        D = _endpoints_nan(np.full(ALPHAS.size, 0.4))
        path = evolve(ALPHAS, D, alpha0=0.9)
        assert np.all(np.diff(path) <= 1e-12)

    def test_stays_inside_the_unit_interval(self):
        D = _endpoints_nan(np.full(ALPHAS.size, -50.0))
        path = evolve(ALPHAS, D, alpha0=0.5, dt=0.5)
        assert path.min() >= 0.0 and path.max() <= 1.0


class TestSettle:
    def test_gain_is_zero_when_evolution_lands_on_anarchy(self):
        D = _endpoints_nan(np.full(ALPHAS.size, 0.4))
        C = 4.0 - 0.2 * ALPHAS
        r = settle(ALPHAS, D, C, alpha0=0.7)
        assert r.alpha_star == pytest.approx(0.0, abs=1e-6)
        assert r.gain == pytest.approx(0.0, abs=1e-6)

    def test_gain_is_one_when_evolution_lands_on_the_optimum(self):
        D = _endpoints_nan(np.full(ALPHAS.size, -0.4))
        C = 4.0 - 0.2 * ALPHAS
        r = settle(ALPHAS, D, C, alpha0=0.3)
        assert r.alpha_star == pytest.approx(1.0, abs=1e-6)
        assert r.gain == pytest.approx(1.0, abs=1e-6)

    def test_interior_attractor_captures_part_of_the_gain(self):
        D = _endpoints_nan(ALPHAS - 0.5)
        C = 4.0 - 0.2 * ALPHAS
        r = settle(ALPHAS, D, C, alpha0=0.2)
        assert r.alpha_star == pytest.approx(0.5, abs=1e-3)
        assert 0.4 < r.gain < 0.6


class TestPigou:
    """The exact two-road solution is the one case we know analytically."""

    def test_altruists_are_always_the_slower_class(self):
        c = pigou_curves(np.linspace(0.0, 1.0, 41))
        D = c["cost_altruist"] - c["cost_selfish"]
        assert np.nanmin(D) > 0.0

    def test_so_imitation_wipes_altruism_out_entirely(self):
        c = pigou_curves(np.linspace(0.0, 1.0, 41))
        D = c["cost_altruist"] - c["cost_selfish"]
        r = settle(c["alpha"], D, c["total_cost"], alpha0=0.9)
        # The Pigou gap closes linearly at the boundary, so the velocity there
        # is quadratic and altruism decays like 1/t rather than exponentially.
        assert r.alpha_star == pytest.approx(0.0, abs=1e-3)
        assert r.gain == pytest.approx(0.0, abs=1e-3)

    def test_a_little_mutation_buys_back_a_little_of_the_gain(self):
        c = pigou_curves(np.linspace(0.0, 1.0, 41))
        D = c["cost_altruist"] - c["cost_selfish"]
        r = settle(c["alpha"], D, c["total_cost"], alpha0=0.9, mu=0.05)
        assert 0.0 < r.alpha_star < 0.5
        assert 0.0 < r.gain < 1.0
        assert pigou_network() is not None
