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

from poi import PC_SQUARE, solve_single_class, square_lattice, solve_mixed  # noqa: E402
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


def _subst_job(args):
    """One disorder realisation: C over the (omega, gamma) grid, plus its optimum."""
    from poi.ignorance import perceive
    from poi.qp import solve_uniform
    from poi import solve_single_class

    L, p, seed, omegas, gammas = args
    net = square_lattice(L, float(p), rng=seed)
    C = np.empty((omegas.size, gammas.size))
    for j, w in enumerate(omegas):
        q = perceive(net, float(w))
        for k, gm in enumerate(gammas):
            C[j, k] = net.total_cost(solve_uniform(q, float(gm)))
    return p, seed, C, solve_single_class(net, "optimum")[1]


def substitution_curve():
    """Numerically locate the cost-minimising uniform altruism gamma*(omega).

    The analytic prediction is gamma* = (2 - 3 omega) / (2 - omega): altruism and
    ignorance are substitutes, and past omega = 2/3 the optimal gamma is negative.
    """
    _banner("Altruism-ignorance substitution curve")
    from concurrent.futures import ProcessPoolExecutor

    ps = np.array([0.30, 0.45, PC_SQUARE, 0.85])
    omegas = np.linspace(0.0, 0.9, 19)
    gammas = np.linspace(-0.8, 1.4, 45)
    n_seeds, L = 24, 16
    C = np.zeros((ps.size, omegas.size, gammas.size))
    Copt = np.zeros(ps.size)
    idx = {float(v): i for i, v in enumerate(ps)}
    jobs = [(L, float(p), s, omegas, gammas) for p in ps for s in range(n_seeds)]
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers()) as ex:
        for p, seed, c, copt in ex.map(_subst_job, jobs, chunksize=1):
            i = idx[p]
            C[i] += c / n_seeds
            Copt[i] += copt / n_seeds
            done += 1
            if done % 8 == 0:
                print(f"  [{done}/{len(jobs)}]", flush=True)
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


def _instr_job(args):
    """One realisation: baseline cost and the effect of each small altruism dose."""
    from poi.ignorance import perceive, solve_ignorant
    from poi.qp import solve_uniform

    L, p, seed, omegas, step = args
    net = square_lattice(L, float(p), rng=seed)
    C0 = np.empty(omegas.size)
    dU = np.empty(omegas.size)
    dF = np.empty(omegas.size)
    for j, w in enumerate(omegas):
        q = perceive(net, float(w))
        C0[j] = net.total_cost(solve_uniform(q, 0.0))
        dU[j] = net.total_cost(solve_uniform(q, step)) - C0[j]
        dF[j] = solve_ignorant(net, float(w), step).total_cost - C0[j]
    return p, C0, dU, dF


def instrument_comparison():
    """Two ways to deliver altruism: a few full altruists, or everyone mildly so.

    The uniform-gamma model has an analytic threshold: the first increment of
    altruism stops helping exactly where gamma*(omega) = 0, i.e. omega = 2/3.
    The fraction-alpha model has no such guarantee, and gives up much earlier --
    concentrating the whole correction on a few drivers overshoots on the paths
    those drivers take.
    """
    _banner("Uniform-gamma versus fraction-alpha as instruments")
    from concurrent.futures import ProcessPoolExecutor

    ps = np.array([0.30, 0.45, PC_SQUARE, 0.85])
    omegas = np.linspace(0.02, 0.94, 47)
    L, n_seeds, step = 18, 32, 0.05
    C0 = np.zeros((ps.size, omegas.size))
    dU = np.zeros((ps.size, omegas.size))
    dF = np.zeros((ps.size, omegas.size))
    idx = {float(v): i for i, v in enumerate(ps)}
    jobs = [(L, float(p), s, omegas, step) for p in ps for s in range(n_seeds)]
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers()) as ex:
        for p, c0, du, df in ex.map(_instr_job, jobs, chunksize=1):
            i = idx[p]
            C0[i] += c0 / n_seeds
            dU[i] += du / n_seeds
            dF[i] += df / n_seeds
            done += 1
            if done % 8 == 0:
                print(f"  [{done}/{len(jobs)}]", flush=True)
    save(
        os.path.join(RESULTS, "instruments.npz"),
        {"p": ps, "omega": omegas, "C0": C0, "dU": dU, "dF": dF,
         "L": L, "n_seeds": n_seeds, "step": step, "pc": PC_SQUARE},
    )


ALL["instruments"] = instrument_comparison


# ===================== real road networks + robustness =====================

TNTP_BASE = "/workspace/bstabler/transportationnetworks"
REAL_NETS = ("SiouxFalls", "Eastern-Massachusetts", "Anaheim")


def _real_job(args):
    """One (network, omega, alpha) point on a real road network."""
    from poi.bpr import frank_wolfe_mixed, perceive_bpr, read_tntp

    name, omega, alpha, tol, max_iter = args
    net = read_tntp(os.path.join(TNTP_BASE, name))
    percv = net if omega == 0.0 else perceive_bpr(net, omega)
    r = frank_wolfe_mixed(percv, alpha, true_net=net, tol=tol, max_iter=max_iter)
    return name, omega, alpha, r.total_cost, r.cost_altruist, r.cost_selfish, r.rel_gap


def real_networks():
    """The (omega, alpha) surface on real road networks from the TNTP benchmark."""
    _banner("Real road networks -- (omega, alpha) surfaces")
    from concurrent.futures import ProcessPoolExecutor

    if not os.path.isdir(TNTP_BASE):
        print(f"  benchmark data not found at {TNTP_BASE}; skipping", flush=True)
        return
    omegas = np.linspace(0.0, 0.9, 10)
    alphas = np.linspace(0.0, 1.0, 11)
    jobs = [(n, float(w), float(a), 1e-6, 900)
            for n in REAL_NETS for w in omegas for a in alphas]
    idx = {n: i for i, n in enumerate(REAL_NETS)}
    shape = (len(REAL_NETS), omegas.size, alphas.size)
    C = np.full(shape, np.nan)
    CA = np.full(shape, np.nan)
    CS = np.full(shape, np.nan)
    gap = np.full(shape, np.nan)
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers()) as ex:
        for name, w, a, c, ca, cs, g in ex.map(_real_job, jobs, chunksize=1):
            i = idx[name]
            j = int(np.argmin(abs(omegas - w)))
            k = int(np.argmin(abs(alphas - a)))
            C[i, j, k], CA[i, j, k], CS[i, j, k], gap[i, j, k] = c, ca, cs, g
            done += 1
            if done % 20 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  [{done}/{len(jobs)}] {el:6.0f}s eta {el/done*(len(jobs)-done):6.0f}s",
                      flush=True)
    save(
        os.path.join(RESULTS, "real_networks.npz"),
        {"names": np.array(REAL_NETS), "omega": omegas, "alpha": alphas,
         "C": C, "CA": CA, "CS": CS, "rel_gap": gap},
    )


def irregular_robustness():
    """Repeat the lattice study on a network with unequal path lengths."""
    _banner("Robustness -- unequal path lengths (skip-DAG)")
    from poi.ignorance import solve_ignorant
    from poi.irregular import path_length_spread, skip_dag

    ps = np.linspace(0.15, 0.9, 16)
    omegas = np.linspace(0.0, 0.9, 10)
    alphas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    n_seeds, K, W = 16, 24, 12
    C = np.zeros((ps.size, omegas.size, alphas.size))
    Copt = np.zeros(ps.size)
    spread = np.zeros((ps.size, 2))
    for i, p in enumerate(ps):
        for s in range(n_seeds):
            net = skip_dag(K, W, float(p), rng=s, skip_frac=0.5)
            if s == 0:
                spread[i] = path_length_spread(net)
            Copt[i] += solve_single_class(net, "optimum")[1] / n_seeds
            for j, w in enumerate(omegas):
                for k, a in enumerate(alphas):
                    C[i, j, k] += solve_ignorant(net, float(w), float(a)).total_cost / n_seeds
        print(f"  p = {p:.3f} done", flush=True)
    save(
        os.path.join(RESULTS, "irregular.npz"),
        {"p": ps, "omega": omegas, "alpha": alphas, "C": C, "Copt": Copt,
         "path_len": spread, "n_seeds": n_seeds, "K": K, "W": W, "pc": PC_SQUARE},
    )


ALL.update({"real_networks": real_networks, "irregular": irregular_robustness})


# ================= idiosyncratic uncertainty: (sigma, rho) ==================

def _stoch_job(args):
    """One lattice realisation over the (sigma, rho) grid, at several alphas."""
    from poi.stochastic import solve_stochastic

    L, p, seed, sigmas, rhos, alphas, K = args
    net = square_lattice(L, float(p), rng=seed)
    C = np.empty((sigmas.size, rhos.size, alphas.size))
    for i, sg in enumerate(sigmas):
        for j, rh in enumerate(rhos):
            for k, al in enumerate(alphas):
                C[i, j, k] = solve_stochastic(
                    net, float(sg), float(rh), K, alpha=float(al), rng=10_000 + seed
                ).total_cost
    return seed, C, solve_single_class(net, "optimum")[1]


def stochastic_lattice():
    """The (sigma, rho) plane on the lattice, crossed with the altruistic fraction."""
    _banner("Idiosyncratic uncertainty -- (sigma, rho) on the lattice")
    from concurrent.futures import ProcessPoolExecutor

    L, K, n_seeds, p = 8, 48, 24, PC_SQUARE
    sigmas = np.array([0.0, 0.02, 0.05, 0.1, 0.15, 0.22, 0.3, 0.45, 0.7, 1.0])
    rhos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    alphas = np.array([0.0, 0.5, 1.0])
    C = np.zeros((sigmas.size, rhos.size, alphas.size))
    Copt = 0.0
    jobs = [(L, p, s, sigmas, rhos, alphas, K) for s in range(n_seeds)]
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers()) as ex:
        for seed, c, copt in ex.map(_stoch_job, jobs, chunksize=1):
            C += c / n_seeds
            Copt += copt / n_seeds
            done += 1
            el = time.time() - t0
            print(f"  [{done}/{len(jobs)}] {el:6.0f}s eta {el/done*(len(jobs)-done):6.0f}s",
                  flush=True)
    save(os.path.join(RESULTS, "stochastic_lattice.npz"),
         {"sigma": sigmas, "rho": rhos, "alpha": alphas, "C": C, "Copt": Copt,
          "L": L, "K": K, "n_seeds": n_seeds, "p": p})


def _kconv_job(args):
    from poi.stochastic import solve_stochastic

    L, p, seed, Ks, sigma, rho = args
    net = square_lattice(L, float(p), rng=seed)
    base = solve_stochastic(net, 0.0, 0.0, 1).total_cost
    return np.array([solve_stochastic(net, sigma, rho, int(K), rng=500 + seed).total_cost
                     for K in Ks]) / base


def stochastic_convergence():
    """How many driver types before the population reads as genuinely idiosyncratic?"""
    _banner("Type-count convergence")
    from concurrent.futures import ProcessPoolExecutor

    Ks = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    L, p, n_seeds = 8, PC_SQUARE, 16
    out = {}
    for rho, tag in [(0.0, "rho0"), (0.5, "rho05")]:
        acc = np.zeros(Ks.size)
        jobs = [(L, p, s, Ks, 0.10, rho) for s in range(n_seeds)]
        with ProcessPoolExecutor(max_workers=n_workers()) as ex:
            for r in ex.map(_kconv_job, jobs, chunksize=1):
                acc += r / n_seeds
        out[tag] = acc
        print(f"  rho={rho}: " + " ".join(f"{v:.4f}" for v in acc), flush=True)
    # The exact rho = 0 continuum, from Dial's algorithm (Gumbel path errors).
    from poi.stochastic import dial_logit
    thetas = np.array([300.0, 100.0, 30.0, 15.0, 10.0, 6.0, 3.0, 1.5, 0.7])
    dial = np.zeros(thetas.size)
    for s in range(n_seeds):
        net = square_lattice(L, p, rng=s)
        eq = solve_single_class(net, "equilibrium")[1]
        for i, th in enumerate(thetas):
            dial[i] += dial_logit(net, float(th), max_iter=2000, tol=1e-12)[1] / eq / n_seeds
    save(os.path.join(RESULTS, "stochastic_convergence.npz"),
         {"K": Ks, "theta": thetas, "dial": dial, "L": L, "p": p,
          "n_seeds": n_seeds, "sigma": 0.10, **out})


def _irr_stoch_job(args):
    from poi.irregular import skip_dag
    from poi.stochastic import solve_stochastic

    K_types, p, seed, sigmas, rhos = args
    net = skip_dag(24, 10, float(p), rng=seed, skip_frac=0.5)
    base = solve_stochastic(net, 0.0, 0.0, 1).total_cost
    C = np.empty((sigmas.size, rhos.size))
    for i, sg in enumerate(sigmas):
        for j, rh in enumerate(rhos):
            C[i, j] = solve_stochastic(net, float(sg), float(rh), K_types,
                                       rng=2000 + seed).total_cost
    return C / base


def stochastic_irregular():
    """Re-run the unequal-path-length case with enough types to be trustworthy."""
    _banner("Idiosyncratic uncertainty on unequal path lengths")
    from concurrent.futures import ProcessPoolExecutor

    sigmas = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0])
    rhos = np.array([0.0, 1.0])
    ps = np.array([0.3, 0.6, 0.9])
    n_seeds, K = 16, 48
    C = np.zeros((ps.size, sigmas.size, rhos.size))
    for ip, p in enumerate(ps):
        jobs = [(K, float(p), s, sigmas, rhos) for s in range(n_seeds)]
        with ProcessPoolExecutor(max_workers=n_workers()) as ex:
            for r in ex.map(_irr_stoch_job, jobs, chunksize=1):
                C[ip] += r / n_seeds
        print(f"  p = {p} done", flush=True)
    save(os.path.join(RESULTS, "stochastic_irregular.npz"),
         {"p": ps, "sigma": sigmas, "rho": rhos, "C": C, "K": K, "n_seeds": n_seeds})


def _real_stoch_job(args):
    from poi.bpr import frank_wolfe_mixed, read_tntp
    from poi.stochastic import solve_stochastic_bpr

    name, sigma, rho, alpha, K = args
    net = read_tntp(os.path.join(TNTP_BASE, name))
    if sigma == 0.0:
        r = frank_wolfe_mixed(net, alpha, tol=1e-6, max_iter=800)
    else:
        r = solve_stochastic_bpr(net, sigma, rho, K, alpha, rng=77,
                                 tol=1e-6, max_iter=800)
    return name, sigma, rho, alpha, r.total_cost, r.rel_gap


def stochastic_real():
    """Does independent error help on a real road network, where bias did not?"""
    _banner("Idiosyncratic uncertainty on real road networks")
    from concurrent.futures import ProcessPoolExecutor

    if not os.path.isdir(TNTP_BASE):
        print("  benchmark data not found; skipping", flush=True)
        return
    names = ("SiouxFalls", "Eastern-Massachusetts")
    sigmas = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0])
    rhos = np.array([0.0, 1.0])
    alphas = np.array([0.0, 1.0])
    K = 12
    jobs = [(n, float(sg), float(rh), float(al), K)
            for n in names for sg in sigmas for rh in rhos for al in alphas]
    idx = {n: i for i, n in enumerate(names)}
    C = np.full((len(names), sigmas.size, rhos.size, alphas.size), np.nan)
    G = np.full_like(C, np.nan)
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers()) as ex:
        for name, sg, rh, al, c, g in ex.map(_real_stoch_job, jobs, chunksize=1):
            i = idx[name]
            j = int(np.argmin(abs(sigmas - sg)))
            k = int(np.argmin(abs(rhos - rh)))
            m = int(np.argmin(abs(alphas - al)))
            C[i, j, k, m], G[i, j, k, m] = c, g
            done += 1
            el = time.time() - t0
            print(f"  [{done}/{len(jobs)}] {el:6.0f}s eta {el/done*(len(jobs)-done):6.0f}s",
                  flush=True)
    save(os.path.join(RESULTS, "stochastic_real.npz"),
         {"names": np.array(names), "sigma": sigmas, "rho": rhos, "alpha": alphas,
          "C": C, "rel_gap": G, "K": K})


def _evo_job(args):
    """One lattice realisation: the (sigma, rho, alpha) cube of C, C_A and C_S."""
    from poi.stochastic import solve_stochastic

    L, p, seed, sigmas, rhos, alphas, K = args
    net = square_lattice(L, float(p), rng=seed)
    shape = (sigmas.size, rhos.size, alphas.size)
    C = np.empty(shape)
    CA = np.empty(shape)
    CS = np.empty(shape)
    for i, sg in enumerate(sigmas):
        for j, rh in enumerate(rhos):
            if sg == 0.0 and j > 0:
                # Zero-magnitude error does not depend on its correlation.
                C[i, j], CA[i, j], CS[i, j] = C[i, 0], CA[i, 0], CS[i, 0]
                continue
            for k, al in enumerate(alphas):
                r = solve_stochastic(net, float(sg), float(rh), K,
                                     alpha=float(al), rng=10_000 + seed)
                C[i, j, k] = r.total_cost
                CA[i, j, k] = r.cost_altruist
                CS[i, j, k] = r.cost_selfish
    return seed, C, CA, CS


def evolution_stochastic():
    """Class costs across the (sigma, rho) plane -- the input to the dynamics.

    :mod:`poi.evolution` needs ``C_A - C_S`` as a function of ``alpha``, not just
    the average cost, because imitation is driven by what each class personally
    experiences.  ``joint_surface`` already supplies that for *systematic*
    ignorance; this run supplies it for idiosyncratic error.
    """
    _banner("Evolution of altruism -- class costs on the (sigma, rho) plane")
    from concurrent.futures import ProcessPoolExecutor

    L, K, n_seeds, p = 8, 48, 12, PC_SQUARE
    sigmas = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.7])
    rhos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    # alpha = 0 and 1 leave one class empty, so the gap is only defined strictly
    # inside; poi.evolution extrapolates to the endpoints from the two nearest
    # samples, which is exactly the "lone altruist" reading.
    alphas = np.linspace(0.05, 0.95, 9)
    shape = (sigmas.size, rhos.size, alphas.size, n_seeds)
    C = np.full(shape, np.nan)
    CA = np.full(shape, np.nan)
    CS = np.full(shape, np.nan)
    jobs = [(L, p, s, sigmas, rhos, alphas, K) for s in range(n_seeds)]
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers()) as ex:
        for seed, c, ca, cs in ex.map(_evo_job, jobs, chunksize=1):
            C[..., seed], CA[..., seed], CS[..., seed] = c, ca, cs
            done += 1
            el = time.time() - t0
            print(f"  [{done}/{len(jobs)}] {el:6.0f}s eta {el/done*(len(jobs)-done):6.0f}s",
                  flush=True)
    save(os.path.join(RESULTS, "evolution_stochastic.npz"),
         {"sigma": sigmas, "rho": rhos, "alpha": alphas, "C": C, "CA": CA, "CS": CS,
          "L": L, "K": K, "n_seeds": n_seeds, "p": p})


def _evo_real_job(args):
    from poi.bpr import frank_wolfe_mixed, read_tntp
    from poi.stochastic import solve_stochastic_bpr

    name, sigma, rho, alpha, K = args
    net = read_tntp(os.path.join(TNTP_BASE, name))
    if sigma == 0.0:
        r = frank_wolfe_mixed(net, alpha, tol=1e-6, max_iter=800)
    else:
        r = solve_stochastic_bpr(net, sigma, rho, K, alpha, rng=77,
                                 tol=1e-6, max_iter=800)
    return name, sigma, rho, alpha, r.total_cost, r.cost_altruist, r.cost_selfish, r.rel_gap


def evolution_real():
    """The same class costs on real road networks under idiosyncratic error."""
    _banner("Evolution of altruism -- class costs on real road networks")
    from concurrent.futures import ProcessPoolExecutor

    if not os.path.isdir(TNTP_BASE):
        print("  benchmark data not found; skipping", flush=True)
        return
    names = ("SiouxFalls", "Eastern-Massachusetts")
    sigmas = np.array([0.0, 0.1, 0.2, 0.4, 0.7])
    rhos = np.array([0.0, 1.0])
    alphas = np.linspace(0.1, 0.9, 5)
    K = 12
    jobs = [(n, float(sg), float(rh), float(al), K)
            for n in names for sg in sigmas for rh in rhos for al in alphas]
    idx = {n: i for i, n in enumerate(names)}
    shape = (len(names), sigmas.size, rhos.size, alphas.size)
    C = np.full(shape, np.nan)
    CA = np.full(shape, np.nan)
    CS = np.full(shape, np.nan)
    G = np.full(shape, np.nan)
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers()) as ex:
        for name, sg, rh, al, c, ca, cs, g in ex.map(_evo_real_job, jobs, chunksize=1):
            i = idx[name]
            j = int(np.argmin(abs(sigmas - sg)))
            k = int(np.argmin(abs(rhos - rh)))
            m = int(np.argmin(abs(alphas - al)))
            C[i, j, k, m], CA[i, j, k, m], CS[i, j, k, m], G[i, j, k, m] = c, ca, cs, g
            done += 1
            el = time.time() - t0
            print(f"  [{done}/{len(jobs)}] {el:6.0f}s eta {el/done*(len(jobs)-done):6.0f}s",
                  flush=True)
    save(os.path.join(RESULTS, "evolution_real.npz"),
         {"names": np.array(names), "sigma": sigmas, "rho": rhos, "alpha": alphas,
          "C": C, "CA": CA, "CS": CS, "rel_gap": G, "K": K})


ALL.update({
    "evolution_stochastic": evolution_stochastic,
    "evolution_real": evolution_real,
    "stochastic_lattice": stochastic_lattice,
    "stochastic_convergence": stochastic_convergence,
    "stochastic_irregular": stochastic_irregular,
    "stochastic_real": stochastic_real,
})


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    which = sys.argv[1:] or list(ALL)
    print(f"workers: {n_workers()}", flush=True)
    t0 = time.time()
    for name in which:
        ALL[name]()
    print(f"\ntotal {time.time() - t0:.1f}s", flush=True)
