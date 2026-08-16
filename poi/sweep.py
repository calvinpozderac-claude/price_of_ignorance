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

from .lattice import square_lattice, bcc_lattice, uniform_entry_demand
from .metrics import summarise_curve
from .qp import solve_mixed

__all__ = ["alpha_sweep", "run_grid", "save", "load", "n_workers"]


def n_workers() -> int:
    return max(1, (os.cpu_count() or 2))


def _build(lattice: str, L: int, p: float, seed: int, entry: str):
    if lattice == "square":
        return square_lattice(L, p, rng=seed, entry=entry)
    if lattice == "bcc":
        return bcc_lattice(L, p, rng=seed)
    raise ValueError(f"unknown lattice {lattice!r}")


def alpha_sweep(net, alphas, demand=None) -> dict:
    """Solve one fixed network at every altruistic fraction in ``alphas``."""
    alphas = np.asarray(alphas, dtype=float)
    out = {k: np.empty(alphas.size) for k in ("C", "CA", "CS", "mu_S", "mu_A", "resid")}
    ok = True
    for i, al in enumerate(alphas):
        s = solve_mixed(net, al, demand=demand)
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
