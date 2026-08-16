"""Print the headline numbers behind every figure.  Run:  python experiments/summary.py"""

from __future__ import annotations

import os
import sys

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


if __name__ == "__main__":
    pigou()
    if os.path.exists(os.path.join(RESULTS, "paper_poa.npz")):
        paper()
    if os.path.exists(os.path.join(RESULTS, "altruism_fine.npz")):
        altruism()
    if os.path.exists(os.path.join(RESULTS, "backfire.npz")):
        backfire()
