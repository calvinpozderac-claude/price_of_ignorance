"""Generate every dataset used by the figures.  Run:  python experiments/compute.py

Datasets land in ``results/`` as compressed ``.npz``:

``paper_poa.npz``
    Price of anarchy versus ``p`` for several system sizes -- the reproduction
    of the paper's Fig. 3.
``paper_scaling.npz``
    Commute time versus ``L`` at ``p < pc``, ``p = pc`` and ``p > pc`` -- Fig. 7.
``altruism_grid.npz``
    The main experiment: the full ``(p, alpha)`` grid of costs and the commute
    times experienced separately by altruistic and selfish drivers.
``altruism_fine.npz``
    A denser ``alpha`` sweep at three representative ``p`` values.
``backfire.npz``
    A wide disorder survey recording where adding altruists makes things worse.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poi import PC_SQUARE, square_lattice, solve_mixed  # noqa: E402
from poi.sweep import n_workers, run_grid, run_grid3, save  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _banner(name):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)


def paper_reproduction():
    """Fig. 3: POA(p) for L = 10, 20, 30, 40; the peak should sit at pc."""
    _banner("Paper Fig. 3 -- POA versus p")
    ps = np.linspace(0.0, 1.0, 41)
    out = {}
    for L, n_seeds in [(10, 96), (20, 64), (30, 40), (40, 24)]:
        print(f"L = {L} ({n_seeds} disorder realisations)", flush=True)
        g = run_grid(ps, np.array([0.0, 1.0]), L=L, n_seeds=n_seeds)
        out[f"C_L{L}"] = g["C"]
        out[f"poa_L{L}"] = g["summary_poa"]
        out[f"nseeds_L{L}"] = n_seeds
    out["p"] = ps
    out["Ls"] = np.array([10, 20, 30, 40])
    out["pc"] = PC_SQUARE
    save(os.path.join(RESULTS, "paper_poa.npz"), out)


def paper_scaling():
    """Fig. 7: C(L) at p = 0.3, pc and 0.8, testing C ~ L, L^m, L^0."""
    _banner("Paper Fig. 7 -- commute time versus system size")
    Ls = np.array([5, 8, 12, 16, 24, 32, 48, 64])
    ps = np.array([0.3, PC_SQUARE, 0.8])
    Ceq = np.full((ps.size, Ls.size), np.nan)
    Copt = np.full((ps.size, Ls.size), np.nan)
    for i, p in enumerate(ps):
        for j, L in enumerate(Ls):
            n = max(4, int(round(400 / L)))
            g = run_grid([p], np.array([0.0, 1.0]), L=int(L), n_seeds=n, progress=False)
            Ceq[i, j] = np.nanmean(g["C"][0, :, 0])
            Copt[i, j] = np.nanmean(g["C"][0, :, 1])
            print(f"  p={p:.4f} L={L:3d}  Ceq={Ceq[i,j]:8.4f}  Copt={Copt[i,j]:8.4f}", flush=True)
    save(
        os.path.join(RESULTS, "paper_scaling.npz"),
        {"L": Ls, "p": ps, "Ceq": Ceq, "Copt": Copt, "pc": PC_SQUARE},
    )


def altruism_grid():
    """Main experiment: cost and per-class experience across the (p, alpha) plane."""
    _banner("Main experiment -- (p, alpha) grid")
    ps = np.linspace(0.05, 0.95, 31)
    alphas = np.linspace(0.0, 1.0, 21)
    g = run_grid(ps, alphas, L=20, n_seeds=48)
    g["pc"] = PC_SQUARE
    save(os.path.join(RESULTS, "altruism_grid.npz"), g)
    print(f"elapsed {g['elapsed']:.1f}s", flush=True)


def altruism_fine():
    """Dense alpha sweeps below, at and above the percolation threshold."""
    _banner("Dense alpha sweeps at three p values")
    ps = np.array([0.45, PC_SQUARE, 0.85])
    alphas = np.linspace(0.0, 1.0, 81)
    g = run_grid(ps, alphas, L=20, n_seeds=64)
    g["pc"] = PC_SQUARE
    save(os.path.join(RESULTS, "altruism_fine.npz"), g)


def backfire_survey():
    """How often, and where, does converting selfish drivers to altruists hurt?"""
    _banner("Backfire survey")
    ps = np.linspace(0.05, 0.95, 31)
    alphas = np.linspace(0.0, 1.0, 41)
    g = run_grid(ps, alphas, L=12, n_seeds=96)
    g["pc"] = PC_SQUARE
    save(os.path.join(RESULTS, "backfire.npz"), g)


def traffic_maps():
    """Per-class flow fields on one network at pc, for the spatial figure."""
    _banner("Traffic maps")
    L = 30
    net = square_lattice(L, PC_SQUARE, rng=7)
    out = {
        "L": L,
        "p": PC_SQUARE,
        "tail": net.tail,
        "head": net.head,
        "pos": net.pos,
        "congestible": net.meta["congestible_mask"],
        "n_lattice_edges": net.meta["n_lattice_edges"],
    }
    for alpha in (0.0, 0.25, 0.5, 1.0):
        s = solve_mixed(net, alpha)
        out[f"x_{alpha}"] = s.x
        out[f"fA_{alpha}"] = s.f_altruist
        out[f"fS_{alpha}"] = s.f_selfish
        out[f"C_{alpha}"] = s.total_cost
        print(f"  alpha={alpha}  C={s.total_cost:.5f}", flush=True)
    save(os.path.join(RESULTS, "traffic_maps.npz"), out)


ALL = {
    "paper_poa": paper_reproduction,
    "paper_scaling": paper_scaling,
    "altruism_grid": altruism_grid,
    "altruism_fine": altruism_fine,
    "backfire": backfire_survey,
    "traffic_maps": traffic_maps,
}




# ============================ user-ignorance extension =====================
# arXiv:2503.09684 crossed with the altruism model.  Note the notation: omega is
# the ignorance level (that paper's alpha); alpha stays the altruistic fraction.

def ignorance_reproduction():
    """Reproduce the ignorance paper: the PI(p, omega) surface at alpha = 0."""
    _banner("Ignorance paper -- price of ignorance surface")
    ps = np.linspace(0.02, 0.98, 33)
    omegas = np.linspace(0.0, 1.0, 26)
    g = run_grid3(ps, omegas, np.array([0.0]), L=20, n_seeds=32)
    g["pc"] = PC_SQUARE
    save(os.path.join(RESULTS, "ignorance_surface.npz"), g)


def ignorance_sizes():
    """PI(p) at omega = 2/3 for several L -- the ignorance paper's Fig. 4(a)."""
    _banner("Ignorance paper -- PI(p) at omega = 2/3 versus system size")
    ps = np.linspace(0.05, 0.95, 31)
    out = {"p": ps, "pc": PC_SQUARE}
    for L, n in [(10, 64), (20, 40), (30, 24), (40, 16)]:
        print(f"L = {L}", flush=True)
        g = run_grid3(ps, np.array([0.0, 2.0 / 3.0]), np.array([0.0]), L=L, n_seeds=n)
        out[f"PI_L{L}"] = g["C"][:, 1, :, 0] / g["C"][:, 0, :, 0]
    out["Ls"] = np.array([10, 20, 30, 40])
    save(os.path.join(RESULTS, "ignorance_sizes.npz"), out)


def joint_surface():
    """The main new experiment: true cost across the (omega, alpha) plane."""
    _banner("Joint (omega, alpha) surface at three p values")
    ps = np.array([0.45, PC_SQUARE, 0.85])
    omegas = np.linspace(0.0, 0.95, 20)
    alphas = np.linspace(0.0, 1.0, 21)
    g = run_grid3(ps, omegas, alphas, L=20, n_seeds=32)
    g["pc"] = PC_SQUARE
    save(os.path.join(RESULTS, "joint_surface.npz"), g)


def altruism_sign_boundary():
    """Where in (p, omega) does adding altruists start to *hurt*?"""
    _banner("Sign boundary of dC/dalpha in the (p, omega) plane")
    ps = np.linspace(0.05, 0.95, 25)
    omegas = np.linspace(0.0, 0.9, 19)
    alphas = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    g = run_grid3(ps, omegas, alphas, L=14, n_seeds=40)
    g["pc"] = PC_SQUARE
    save(os.path.join(RESULTS, "sign_boundary.npz"), g)


def substitution_curve():
    """Numerically locate the cost-minimising uniform altruism gamma*(omega).

    The analytic prediction is gamma* = (2 - 3 omega) / (2 - omega): altruism and
    ignorance are substitutes, and past omega = 2/3 the optimal gamma is negative.
    """
    _banner("Altruism-ignorance substitution curve")
    from poi.ignorance import perceive
    from poi.qp import solve_uniform
    from poi import solve_single_class

    ps = np.array([0.30, 0.45, PC_SQUARE, 0.85])
    omegas = np.linspace(0.0, 0.9, 19)
    gammas = np.linspace(-0.8, 1.4, 111)
    n_seeds = 16
    L = 20
    C = np.zeros((ps.size, omegas.size, gammas.size))
    Copt = np.zeros(ps.size)
    for i, p in enumerate(ps):
        for seed in range(n_seeds):
            net = square_lattice(L, float(p), rng=seed)
            Copt[i] += solve_single_class(net, "optimum")[1] / n_seeds
            for j, w in enumerate(omegas):
                q = perceive(net, float(w))
                for k, gm in enumerate(gammas):
                    C[i, j, k] += net.total_cost(solve_uniform(q, float(gm))) / n_seeds
        print(f"  p = {p:.4f} done", flush=True)
    save(
        os.path.join(RESULTS, "substitution.npz"),
        {"p": ps, "omega": omegas, "gamma": gammas, "C": C, "Copt": Copt,
         "L": L, "n_seeds": n_seeds, "pc": PC_SQUARE},
    )


ALL.update({
    "ignorance_surface": ignorance_reproduction,
    "ignorance_sizes": ignorance_sizes,
    "joint_surface": joint_surface,
    "sign_boundary": altruism_sign_boundary,
    "substitution": substitution_curve,
})


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    which = sys.argv[1:] or list(ALL)
    print(f"workers: {n_workers()}", flush=True)
    t0 = time.time()
    for name in which:
        ALL[name]()
    print(f"\ntotal {time.time() - t0:.1f}s", flush=True)
