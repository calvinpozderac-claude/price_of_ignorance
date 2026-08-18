"""Parallel sweeps over disorder ``p``, altruistic fraction ``alpha`` and system size.

Every sweep is disorder-averaged: for each ``(L, p)`` a number of independent
random road layouts are drawn, and each layout is solved at every ``alpha``.
Because one layout is reused across all ``alpha`` values, the alpha-dependence
is measured *within* a fixed network, which is what makes curves like
:func:`poi.metrics.saturation_alpha` meaningful.

Results are returned as dictionaries of arrays shaped ``(n_p, n_seeds, n_alpha)``
and are saved with :func:`save` / :func:`load` as compressed ``.npz``.
"""

from __future__ import annotations

import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .ignorance import perceive
from .lattice import square_lattice, bcc_lattice, uniform_entry_demand
from .metrics import summarise_curve
from .qp import solve_mixed

__all__ = ["alpha_sweep", "run_grid", "run_grid3", "save", "load", "n_workers"]


def n_workers() -> int:
    return max(1, (os.cpu_count() or 2))


def _build(lattice: str, L: int, p: float, seed: int, entry: str):
    if lattice == "square":
        return square_lattice(L, p, rng=seed, entry=entry)
    if lattice == "bcc":
        return bcc_lattice(L, p, rng=seed)
    raise ValueError(f"unknown lattice {lattice!r}")


def alpha_sweep(net, alphas, demand=None, omega: float = 0.0) -> dict:
    """Solve one fixed network at every altruistic fraction in ``alphas``.

    ``omega`` is the user-ignorance level: drivers plan against
    ``perceive(net, omega)`` while outcomes are scored with the true ``net``.
    """
    alphas = np.asarray(alphas, dtype=float)
    percv = net if omega == 0.0 else perceive(net, omega)
    out = {k: np.empty(alphas.size) for k in ("C", "CA", "CS", "mu_S", "mu_A", "resid")}
    ok = True
    for i, al in enumerate(alphas):
        s = solve_mixed(percv, al, demand=demand, true_net=net)
        ok &= s.status == "Solved"
        out["C"][i] = s.total_cost
        out["CA"][i] = s.cost_altruist
        out["CS"][i] = s.cost_selfish
        out["mu_S"][i] = s.mu_selfish
        out["mu_A"][i] = s.mu_altruist
        out["resid"][i] = max(s.residual_selfish, s.residual_altruist)
    out["alpha"] = alphas
    out["all_solved"] = ok
    return out


def _task(args):
    lattice, L, p, seed, alphas, entry = args
    net = _build(lattice, L, p, seed, entry)
    demand = uniform_entry_demand(net) if entry == "uniform" else None
    res = alpha_sweep(net, alphas, demand=demand)
    return (p, seed, res)


def run_grid(
    ps,
    alphas,
    L: int = 20,
    n_seeds: int = 16,
    lattice: str = "square",
    entry: str = "busbar",
    seed0: int = 0,
    workers: int | None = None,
    progress: bool = True,
) -> dict:
    """Sweep the full ``(p, seed, alpha)`` grid in parallel.

    Returns a dict with ``C``, ``CA``, ``CS`` of shape ``(n_p, n_seeds, n_alpha)``
    plus the axes and per-``(p, seed)`` scalar summaries from
    :func:`poi.metrics.summarise_curve`.
    """
    ps = np.asarray(ps, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    workers = workers or n_workers()

    jobs = [
        (lattice, L, float(p), seed0 + s, alphas, entry)
        for p, s in itertools.product(ps, range(n_seeds))
    ]
    shape = (ps.size, n_seeds, alphas.size)
    C = np.full(shape, np.nan)
    CA = np.full(shape, np.nan)
    CS = np.full(shape, np.nan)
    resid = np.full(shape, np.nan)
    p_index = {float(p): i for i, p in enumerate(ps)}

    summaries: dict[str, np.ndarray] = {}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for p, seed, res in ex.map(_task, jobs, chunksize=1):
            i, j = p_index[p], seed - seed0
            C[i, j] = res["C"]
            CA[i, j] = res["CA"]
            CS[i, j] = res["CS"]
            resid[i, j] = res["resid"]
            s = summarise_curve(alphas, res["C"], res["CA"], res["CS"])
            for k, v in s.items():
                summaries.setdefault(k, np.full((ps.size, n_seeds), np.nan))[i, j] = v
            done += 1
            if progress and (done % max(1, len(jobs) // 40) == 0 or done == len(jobs)):
                el = time.time() - t0
                print(
                    f"  [{done}/{len(jobs)}] {el:6.1f}s  eta {el/done*(len(jobs)-done):6.1f}s",
                    flush=True,
                )

    return {
        "p": ps,
        "alpha": alphas,
        "L": L,
        "n_seeds": n_seeds,
        "lattice": lattice,
        "entry": entry,
        "C": C,
        "CA": CA,
        "CS": CS,
        "resid": resid,
        "elapsed": time.time() - t0,
        **{f"summary_{k}": v for k, v in summaries.items()},
    }


def save(path, data: dict) -> None:
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in data.items()})


def load(path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _task3(args):
    lattice, L, p, omega, seed, alphas, entry = args
    net = _build(lattice, L, p, seed, entry)
    demand = uniform_entry_demand(net) if entry == "uniform" else None
    return (p, omega, seed, alpha_sweep(net, alphas, demand=demand, omega=omega))


def run_grid3(
    ps,
    omegas,
    alphas,
    L: int = 20,
    n_seeds: int = 24,
    lattice: str = "square",
    entry: str = "busbar",
    seed0: int = 0,
    workers: int | None = None,
    progress: bool = True,
) -> dict:
    """Sweep the ``(p, omega, seed, alpha)`` grid in parallel.

    Returns ``C``, ``CA``, ``CS`` of shape ``(n_p, n_omega, n_seeds, n_alpha)``.
    The same random network is reused across every ``omega`` and ``alpha`` for a
    given ``(p, seed)``, so the two parameter dependences are measured within a
    fixed road layout.
    """
    ps = np.asarray(ps, dtype=float)
    omegas = np.asarray(omegas, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    workers = workers or n_workers()

    jobs = [
        (lattice, L, float(p), float(w), seed0 + s, alphas, entry)
        for p, w, s in itertools.product(ps, omegas, range(n_seeds))
    ]
    shape = (ps.size, omegas.size, n_seeds, alphas.size)
    C = np.full(shape, np.nan)
    CA = np.full(shape, np.nan)
    CS = np.full(shape, np.nan)
    pi = {float(v): i for i, v in enumerate(ps)}
    wi = {float(v): i for i, v in enumerate(omegas)}

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for p, w, seed, res in ex.map(_task3, jobs, chunksize=1):
            i, j, k = pi[p], wi[w], seed - seed0
            C[i, j, k] = res["C"]
            CA[i, j, k] = res["CA"]
            CS[i, j, k] = res["CS"]
            done += 1
            if progress and (done % max(1, len(jobs) // 30) == 0 or done == len(jobs)):
                el = time.time() - t0
                print(
                    f"  [{done}/{len(jobs)}] {el:6.1f}s  eta {el/done*(len(jobs)-done):6.1f}s",
                    flush=True,
                )

    return {
        "p": ps,
        "omega": omegas,
        "alpha": alphas,
        "L": L,
        "n_seeds": n_seeds,
        "lattice": lattice,
        "C": C,
        "CA": CA,
        "CS": CS,
        "elapsed": time.time() - t0,
    }
