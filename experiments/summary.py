"""Print the headline numbers behind every figure.  Run:  python experiments/summary.py"""

from __future__ import annotations

import os
import sys

import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poi import pigou_curves  # noqa: E402
from poi.metrics import efficiency, saturation_alpha  # noqa: E402
from poi.sweep import load  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def head(t):
    print(f"\n{t}\n{'-' * len(t)}")


def pigou():
    head("Pigou's two roads (exact)")
    a = np.linspace(0, 1, 1001)
    c = pigou_curves(a)
    print(f"C_eq = {c['C_eq']:.4f}   C_opt = {c['C_opt']:.4f}   POA = {c['C_eq']/c['C_opt']:.4f}")
    print(f"saturating fraction alpha* = {c['alpha_star']:.3f}")
    for al in (0.1, 0.25, 0.5, 0.75, 1.0):
        i = np.argmin(abs(a - al))
        print(
            f"  alpha={al:4.2f}  C={c['total_cost'][i]:.4f}  C_A={c['cost_altruist'][i]:.4f}"
            f"  C_S={c['cost_selfish'][i]:.4f}  gap={c['gap'][i]:.4f}"
            f"  gain captured={c['efficiency'][i]*100:5.1f}%"
        )
    i25 = np.argmin(abs(a - 0.25))
    print(f"\n  a quarter of drivers altruistic already captures {c['efficiency'][i25]*100:.0f}% of the gain")
    print(f"  peak gap {np.nanmax(c['gap']):.3f} at alpha = {a[np.nanargmax(c['gap'])]:.3f}")


def paper():
    head("Paper reproduction")
    d = load(os.path.join(RESULTS, "paper_poa.npz"))
    p, pc = d["p"], float(d["pc"])
    for L in d["Ls"]:
        m = np.nanmean(d[f"poa_L{L}"], axis=1)
        print(f"  L={int(L):3d}  peak POA = {np.nanmax(m):.4f} at p = {p[np.nanargmax(m)]:.3f}   (pc = {pc})")

    s = load(os.path.join(RESULTS, "paper_scaling.npz"))
    L = s["L"]
    names = ["p = 0.30 (< pc)", "p = pc", "p = 0.80 (> pc)"]
    print("\n  fitted exponent of C ~ L^m (equilibrium):")
    for i, n in enumerate(names):
        m = np.polyfit(np.log(L), np.log(s["Ceq"][i]), 1)[0]
        print(f"    {n:18s}  m = {m:5.2f}")
    print("    paper predicts m = 1, 0.388 and 0 respectively (fitted 0.31 in the paper)")


def altruism():
    head("Mixed population: does altruism help?")
    d = load(os.path.join(RESULTS, "altruism_fine.npz"))
    al, ps = d["alpha"], d["p"]
    for i, p in enumerate(ps):
        C, CA, CS = d["C"][i], d["CA"][i], d["CS"][i]
        mC = np.nanmean(C, axis=0)
        eff = np.nanmean([efficiency(r) for r in C], axis=0)
        astar = np.mean([saturation_alpha(al, r) for r in C])
        gap = np.nanmean(CA - CS, axis=0)
        regret = np.nanmean(CA - C[:, [0]], axis=0)
        j = np.argmin(abs(al - 0.25))
        print(f"\n  p = {p:.4f}   (L = {int(d['L'])}, {int(d['n_seeds'])} realisations)")
        print(f"    POA = C(0)/C(1)          = {mC[0]/mC[-1]:.4f}")
        print(f"    mean saturating alpha*   = {astar:.3f}")
        print(f"    gain captured at a=0.25  = {eff[j]*100:.1f}%")
        print(f"    gain captured at a=0.50  = {eff[np.argmin(abs(al-0.5))]*100:.1f}%")
        print(f"    peak altruist/selfish gap= {np.nanmax(gap):.4f} at alpha = {al[np.nanargmax(gap)]:.2f}")
        print(f"      ...as % of the optimum = {100*np.nanmax(gap)/mC[-1]:.1f}%")
        worse = np.isfinite(regret) & (regret > 0)
        print(f"    altruists worse off than full anarchy over {100*worse.mean():.0f}% of the alpha range")
        print(f"      worst by                = {np.nanmax(regret):.4f} ({100*np.nanmax(regret)/mC[0]:.1f}% of C_eq)")
        best = np.nanmin(np.nanmean(CS, axis=0))
        print(f"    best selfish commute     = {best:.4f} vs C_opt = {mC[-1]:.4f} ({100*(1-best/mC[-1]):.0f}% below the social optimum)")


def backfire():
    head("Can adding altruists make things worse?")
    d = load(os.path.join(RESULTS, "backfire.npz"))
    p, mb = d["p"], d["summary_max_backfire"]
    Copt = d["C"][:, :, -1]
    frac = np.mean(mb > 1e-7, axis=1)
    rel = np.nanmean(mb / Copt, axis=1)
    print(f"  overall: {100*np.mean(mb > 1e-7):.1f}% of {mb.size} random networks show a range of alpha")
    print("           over which converting selfish drivers to altruists RAISES the average commute")
    i = np.argmax(frac)
    print(f"  worst p = {p[i]:.3f}: {100*frac[i]:.0f}% of networks, mean worst rise {100*rel[i]:.2f}% of C_opt")
    j = np.argmax(rel)
    print(f"  largest mean rise at p = {p[j]:.3f}: {100*rel[j]:.2f}% of C_opt")
    print(f"  max single-network rise: {100*np.nanmax(mb/Copt):.2f}% of C_opt")

    # The stronger claim: a mixed population that is worse than pure anarchy.
    C = d["C"]
    worse = C > C[:, :, [0]] + 1e-7
    print(
        f"  cases where a mixed population is worse than FULL ANARCHY: "
        f"{100*worse.mean():.2f}% of all (network, alpha) pairs"
    )
    if worse.any():
        ip = np.argmax(worse.reshape(worse.shape[0], -1).mean(axis=1))
        print(f"    concentrated near p = {p[ip]:.3f}")




# ==================== user-ignorance extension (arXiv:2503.09684) ==========
OMEGA_STAR = 2.0 / 3.0


def gamma_star(w):
    return (2 - 3 * np.asarray(w, float)) / (2 - np.asarray(w, float))


def ignorance():
    head("Ignorance paper reproduced (omega = ignorance, their alpha)")
    d = load(os.path.join(RESULTS, "ignorance_surface.npz"))
    p, w = d["p"], d["omega"]
    C = d["C"][:, :, :, 0]
    PI = np.nanmean(C / C[:, [0], :], axis=2)
    below = w <= OMEGA_STAR + 1e-9
    print(f"  max PI over all p, for omega <= 2/3 : {PI[:, below].max():.4f}  (paper: <= 1)")
    i, j = np.unravel_index(np.argmin(PI), PI.shape)
    print(f"  minimum PI = {PI.min():.4f} at p = {p[i]:.3f}, omega = {w[j]:.3f}")
    print(f"    (paper: min ~0.95 near p_c = {float(d['pc'])}, omega ~ 2/3)")
    jj = int(np.argmin(abs(w - OMEGA_STAR)))
    print(f"  at omega = 2/3 exactly: max PI = {PI[:, jj].max():.4f}, min = {PI[:, jj].min():.4f}")
    # limit of useful ignorance
    star = []
    for i in range(p.size):
        over = np.flatnonzero(PI[i] > 1 + 1e-4)
        star.append(w[over[0]] if over.size else 1.0)
    star = np.array(star)
    lo = p < float(d["pc"])
    print(f"  limit of useful ignorance omega*: {star[lo].mean():.3f} below p_c (paper quotes ~6/7 = {6/7:.3f})")


def joint():
    head("Altruism crossed with ignorance")
    d = load(os.path.join(RESULTS, "joint_surface.npz"))
    p, w, al = d["p"], d["omega"], d["alpha"]
    C = np.nanmean(d["C"], axis=2)
    for i, pp in enumerate(p):
        rel = C[i] / C[i].min()
        j, k = np.unravel_index(np.argmin(C[i]), C[i].shape)
        print(f"\n  p = {pp:.4f}   (L = {int(d['L'])}, {int(d['n_seeds'])} realisations)")
        print(f"    cheapest overall at omega = {w[j]:.2f}, alpha = {al[k]:.2f}  -> C = {C[i].min():.4f}")
        print(f"    all-selfish, no ignorance   C = {C[i][0, 0]:.4f}  (+{100*(C[i][0,0]/C[i].min()-1):.2f}%)")
        print(f"    all-altruistic, no ignorance C = {C[i][0, -1]:.4f}  (+{100*(C[i][0,-1]/C[i].min()-1):.2f}%)")
        jj = int(np.argmin(abs(w - OMEGA_STAR)))
        print(f"    ignorant selfish (w=2/3, a=0) C = {C[i][jj, 0]:.4f}  (+{100*(C[i][jj,0]/C[i].min()-1):.2f}%)")
        print(f"    BOTH  (w=2/3, a=1)            C = {C[i][jj, -1]:.4f}  (+{100*(C[i][jj,-1]/C[i].min()-1):.2f}%)")
        print(f"      -> adding altruists at w=2/3 changes C by {100*(C[i][jj,-1]/C[i][jj,0]-1):+.2f}%")
        # where does the sign flip?
        delta = C[i][:, -1] - C[i][:, 0]
        flip = np.flatnonzero(delta > 0)
        if flip.size:
            print(f"    altruism starts hurting at omega ~ {w[flip[0]]:.2f}")


def sign_boundary():
    head("Where does altruism flip from helping to hurting?")
    d = load(os.path.join(RESULTS, "sign_boundary.npz"))
    p, w = d["p"], d["omega"]
    C = np.nanmean(d["C"], axis=2)
    delta = C[:, :, -1] - C[:, :, 0]
    print("    p      omega_c (alpha=1 vs 0)")
    wc = []
    for i in range(p.size):
        f = np.flatnonzero(delta[i] > 0)
        v = w[f[0]] if f.size else np.nan
        wc.append(v)
        if i % 3 == 0:
            print(f"   {p[i]:.3f}    {v:.3f}")
    wc = np.array(wc, float)
    print(f"\n  mean omega_c over all p = {np.nanmean(wc):.3f}   (2/3 = {OMEGA_STAR:.3f})")
    print(f"  spread: {np.nanmin(wc):.3f} to {np.nanmax(wc):.3f}")
    frac = np.mean(delta > 0)
    print(f"  fraction of the (p, omega) grid where altruism makes traffic worse: {100*frac:.0f}%")


def substitution():
    head("The altruism-ignorance substitution curve")
    d = load(os.path.join(RESULTS, "substitution.npz"))
    p, w, g, C = d["p"], d["omega"], d["gamma"], d["C"]
    print("  predicted gamma* = (2-3w)/(2-w);  measured = argmin_gamma C")
    print("   omega   predicted " + "".join(f"  p={v:.2f}" for v in p))
    errs = []
    for j, ww in enumerate(w):
        if j % 3:
            continue
        meas = [g[np.argmin(C[i, j])] for i in range(p.size)]
        errs += [abs(m - gamma_star(ww)) for m in meas]
        print(f"   {ww:.3f}   {gamma_star(ww):+.3f}   " + "".join(f" {m:+.3f}" for m in meas))
    allm = np.array([[g[np.argmin(C[i, j])] for i in range(p.size)] for j in range(w.size)])
    err = np.abs(allm - gamma_star(w)[:, None])
    print(f"\n  max |measured - predicted| = {err.max():.3f}  (gamma grid step {g[1]-g[0]:.3f})")
    print(f"  mean |error|               = {err.mean():.4f}")
    print(f"  gamma* < 0 (spite needed) once omega > 2/3: {gamma_star(0.8):+.3f} at omega=0.8")


def real_networks():
    head("Real road networks (TNTP benchmark, BPR costs, Frank-Wolfe)")
    d = load(os.path.join(RESULTS, "real_networks.npz"))
    names = [str(x) for x in d["names"]]
    w, al, C, g = d["omega"], d["alpha"], d["C"], d["rel_gap"]
    print(f"  Frank-Wolfe relative gap: median {np.nanmedian(g):.1e}, max {np.nanmax(g):.1e}")
    for i, n in enumerate(names):
        pi = C[i, :, 0] / C[i, 0, 0]
        eff = 100 * (C[i, :, -1] / C[i, :, 0] - 1)
        flip = np.flatnonzero(eff > 0)
        print(f"\n  {n}")
        print(f"    price of anarchy                 = {C[i,0,0]/C[i,0,-1]:.4f}")
        print(f"    price of ignorance at omega=2/3  = {np.interp(2/3, w, pi):.4f}"
              f"   (lattice: ~0.95)")
        print(f"    worst price of ignorance         = {pi.max():.4f} at omega = {w[pi.argmax()]:.1f}")
        print(f"    altruism effect at omega=0       = {eff[0]:+.2f}%")
        print(f"    altruism effect at omega=2/3     = {np.interp(2/3, w, eff):+.2f}%"
              f"   (lattice: about +5%)")
        print(f"    altruism flips sign at omega     = "
              f"{w[flip[0]]:.1f}" if flip.size else "    altruism never hurts")
        jb, kb = np.unravel_index(np.nanargmin(C[i]), C[i].shape)
        print(f"    cheapest mix: omega={w[jb]:.1f}, alpha={al[kb]:.1f}")


def irregular():
    head("Robustness: unequal path lengths (skip-DAG)")
    d = load(os.path.join(RESULTS, "irregular.npz"))
    p, w, C = d["p"], d["omega"], d["C"]
    lo, hi = d["path_len"][0].astype(int)
    print(f"  path lengths range {lo}-{hi} roads (lattice: exactly {2*int(d['K'])//2})")
    j = int(np.argmin(abs(w - 2 / 3)))
    pi = C[:, j, 0] / C[:, 0, 0]
    print(f"  price of ignorance at omega=2/3: {pi.min():.3f} to {pi.max():.3f}")
    print(f"    helps (PI<1) for {100*np.mean(pi < 1):.0f}% of p values"
          f"  -- on the lattice it helps for 100%")
    rel = C[:, :, -1] / C[:, :, 0] - 1
    print(f"  altruism hurts over {100*np.mean(rel > 0):.0f}% of the (p, omega) grid"
          f"  (lattice: 57%)")
    flip = []
    for i in range(p.size):
        f = np.flatnonzero(rel[i] > 0)
        flip.append(w[f[0]] if f.size else np.nan)
    print(f"  mean omega where altruism flips = {np.nanmean(flip):.2f}  (lattice: ~0.45-0.50)")
    best = np.unravel_index(np.argmin(rel), rel.shape)
    print(f"  altruism helps MOST at p={p[best[0]]:.2f}, omega={w[best[1]]:.1f}: {100*rel[best]:+.1f}%")
    print("    -- a regime absent from the lattice: where ignorance does damage,")
    print("       altruists repair it instead of over-correcting.")


def stochastic():
    head("Idiosyncratic error: magnitude sigma, correlation rho")
    d = load(os.path.join(RESULTS, "stochastic_lattice.npz"))
    sg, rh, C = d["sigma"], d["rho"], d["C"]
    b = C[0, 0, 0]
    print(f"  lattice L={int(d['L'])} p=pc, K={int(d['K'])} types, {int(d['n_seeds'])} networks")
    print("  best achievable benefit, and where:")
    for j, r in enumerate(rh):
        i = int(np.argmin(C[:, j, 0]))
        print(f"    rho={r:.2f}:  {100*(C[i,j,0]/b-1):+.2f}% at sigma={sg[i]:.2f}"
              f"   |  damage at sigma=1: {100*(C[-1,j,0]/b-1):+.0f}%")
    print("\n  the beneficial window shrinks with correlation and closes at rho=1")

    k = load(os.path.join(RESULTS, "stochastic_convergence.npz"))
    print(f"\n  type-count convergence at sigma={float(k['sigma']):.2f} (% change in C):")
    print("    K:      " + " ".join(f"{int(v):6d}" for v in k["K"]))
    print("    rho=0   " + " ".join(f"{100*(v-1):+6.2f}" for v in k["rho0"]))
    print("    rho=0.5 " + " ".join(f"{100*(v-1):+6.2f}" for v in k["rho05"]))
    print(f"    K=1 is a single shared bias, so it reads as rho=1; the answer only")
    print(f"    settles past K ~ 64.")
    print(f"    Dial logit continuum (Gumbel path errors, exact): "
          f"{100*(k['dial'].min()-1):+.2f}% at theta={k['theta'][int(k['dial'].argmin())]:.0f}")

    print("\n  altruism effect (alpha 0 -> 1), % change in C:")
    print("    sigma:  " + " ".join(f"{v:6.2f}" for v in sg))
    for j in (0, rh.size - 1):
        print(f"    rho={rh[j]:.0f}   " + " ".join(
            f"{100*(C[i,j,-1]/C[i,j,0]-1):+6.2f}" for i in range(sg.size)))
    print("    altruism reverses sign under INDEPENDENT error and helps ever more")
    print("    under SHARED error -- the opposite way round from the prediction,")
    print("    but consistent with the rule: altruism backfires exactly where the")
    print("    uncertainty is already doing corrective work.")

    if os.path.exists(os.path.join(RESULTS, "stochastic_real.npz")):
        r = load(os.path.join(RESULTS, "stochastic_real.npz"))
        names = [str(x) for x in r["names"]]
        sg2, C2 = r["sigma"], r["C"]
        print(f"\n  real road networks (K={int(r['K'])}, max FW gap {np.nanmax(r['rel_gap']):.1e}):")
        for i, n in enumerate(names):
            base = C2[i, 0, 0, 0]
            ind = C2[i, :, 0, 0] / base
            j = int(np.argmin(ind))
            print(f"    {n}: best independent-error benefit {100*(ind[j]-1):+.2f}% at sigma={sg2[j]:.2f}"
                  f"; shared error best {100*(np.min(C2[i,:,1,0]/base)-1):+.2f}%")


def evolution():
    """Study 1: does altruism survive when drivers can copy whoever is faster?"""
    from poi.evolution import invasion_signs, settle

    head("Evolutionary stability: can altruism sustain itself?")
    print("  imitation dynamics  d(alpha)/dt = alpha (1-alpha) (C_S - C_A)")
    print("  D(0) < 0 means a lone altruist in a selfish crowd already arrives sooner.")

    d = load(os.path.join(RESULTS, "joint_surface.npz"))
    p, w, al = d["p"], d["omega"], d["alpha"]
    C, CA, CS = (np.nanmean(d[k], axis=2) for k in ("C", "CA", "CS"))
    print("\n  LATTICE, systematic ignorance -- D(0), the lone altruist's penalty")
    print("    omega:  " + " ".join(f"{v:6.2f}" for v in w[::3]))
    for i in range(p.size):
        row = [invasion_signs(al, CA[i, j] - CS[i, j])[0] for j in range(0, w.size, 3)]
        print(f"    p={p[i]:.3f} " + " ".join(f"{v:+6.2f}" for v in row))
    n_inv = sum(invasion_signs(al, CA[i, j] - CS[i, j])[0] < 0
                for i in range(p.size) for j in range(w.size))
    print(f"    altruism invades in {n_inv} of {p.size * w.size} cells -- it never does,")
    print("    and ignorance makes the altruist's penalty larger, not smaller.")

    if os.path.exists(os.path.join(RESULTS, "real_networks.npz")):
        r = load(os.path.join(RESULTS, "real_networks.npz"))
        names = [str(x) for x in r["names"]]
        w2, al2 = r["omega"], r["alpha"]
        print("\n  REAL NETWORKS, systematic ignorance -- the opposite answer")
        for i, n in enumerate(names):
            print(f"    {n}")
            print("      omega   D(0)     D(1)   settled alpha   gain captured")
            for j in range(0, w2.size, 2):
                D = r["CA"][i, j] - r["CS"][i, j]
                d0, d1 = invasion_signs(al2, D)
                e = settle(al2, D, r["C"][i, j], alpha0=0.5)
                print(f"      {w2[j]:.2f}  {d0:+7.3f}  {d1:+7.3f}   {e.alpha_star:11.3f}"
                      f"   {100*e.gain:11.1f}%")
            first = next((w2[j] for j in range(w2.size)
                          if invasion_signs(al2, r["CA"][i, j] - r["CS"][i, j])[0] < 0), None)
            print(f"      altruism first invades at omega = "
                  f"{'never' if first is None else f'{first:.2f}'}")

    f = os.path.join(RESULTS, "evolution_stochastic.npz")
    if os.path.exists(f):
        e = load(f)
        sg, rh, al3 = e["sigma"], e["rho"], e["alpha"]
        C3, CA3, CS3 = (np.nanmean(e[k], axis=3) for k in ("C", "CA", "CS"))
        print(f"\n  LATTICE, idiosyncratic error (L={int(e['L'])}, K={int(e['K'])}, "
              f"{int(e['n_seeds'])} networks)")
        print("    settled altruistic fraction, from alpha0 = 0.5")
        print("    sigma:   " + " ".join(f"{v:5.2f}" for v in sg))
        for j, rv in enumerate(rh):
            row = [settle(al3, CA3[i, j] - CS3[i, j], C3[i, j], alpha0=0.5).alpha_star
                   for i in range(sg.size)]
            print(f"    rho={rv:.2f} " + " ".join(f"{v:5.2f}" for v in row))
        print("\n    share of the achievable gain the settled population captures")
        print("    sigma:   " + " ".join(f"{v:5.2f}" for v in sg))
        for j, rv in enumerate(rh):
            row = [settle(al3, CA3[i, j] - CS3[i, j], C3[i, j], alpha0=0.5).gain
                   for i in range(sg.size)]
            print(f"    rho={rv:.2f} " + " ".join(f"{100*v:4.0f}%" for v in row))
        best = max(((i, j) for i in range(sg.size) for j in range(rh.size)),
                   key=lambda t: settle(al3, CA3[t[0], t[1]] - CS3[t[0], t[1]],
                                        C3[t[0], t[1]], alpha0=0.5).alpha_star)
        b = settle(al3, CA3[best] - CS3[best], C3[best], alpha0=0.5)
        print(f"\n    highest self-sustaining altruism: alpha* = {b.alpha_star:.2f} at "
              f"sigma={sg[best[0]]:.2f}, rho={rh[best[1]]:.2f}"
              f"  ({100*b.gain:.0f}% of the gain, unenforced)")

    f = os.path.join(RESULTS, "evolution_real.npz")
    if os.path.exists(f):
        e = load(f)
        names = [str(x) for x in e["names"]]
        sg, rh, al4 = e["sigma"], e["rho"], e["alpha"]
        print(f"\n  REAL NETWORKS, idiosyncratic error (K={int(e['K'])}, "
              f"max FW gap {np.nanmax(e['rel_gap']):.1e})")
        for i, n in enumerate(names):
            print(f"    {n}: settled alpha")
            print("      sigma: " + " ".join(f"{v:5.2f}" for v in sg))
            for j, rv in enumerate(rh):
                row = [settle(al4, e["CA"][i, k, j] - e["CS"][i, k, j],
                              e["C"][i, k, j], alpha0=0.5).alpha_star for k in range(sg.size)]
                print(f"      rho={rv:.0f}  " + " ".join(f"{v:5.2f}" for v in row))


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices at alpha 0 and 1
    for name, fn in [
        (None, pigou),
        ("paper_poa", paper),
        ("altruism_fine", altruism),
        ("backfire", backfire),
        ("ignorance_surface", ignorance),
        ("joint_surface", joint),
        ("sign_boundary", sign_boundary),
        ("substitution", substitution),
        ("irregular", irregular),
        ("real_networks", real_networks),
        ("stochastic_lattice", stochastic),
        ("joint_surface", evolution),
    ]:
        if name is None or os.path.exists(os.path.join(RESULTS, name + ".npz")):
            fn()
