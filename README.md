# The price of ignorance

Do altruistic drivers actually speed traffic up, or do they just get exploited by
the greedy ones?

This repo extends Brian Skinner's *[The price of anarchy is maximized at the
percolation threshold](https://arxiv.org/abs/1404.2935)* (arXiv:1404.2935) from a
population that is either entirely selfish or entirely centrally-planned to a
**mixed population**: a fraction `α` of drivers route to minimise the *total*
commute time, and `1 - α` route to minimise their own. Everything is solved as a
convex quadratic program with [Clarabel](https://clarabel.org), plus a closed-form
solution of the two-road case.

The short answer is **both things are true at once, and they are the same
phenomenon**:

* Total commute time really does fall as `α` rises — but the gains are
  front-loaded and saturate at a finite `α* < 1`, past which extra altruists
  accomplish nothing.
* Every unit of that gain is collected by the selfish drivers. Altruists always
  experience a *longer* commute than selfish drivers, and on the lattice they end
  up worse off than **everybody** was under pure anarchy.
* The percolation threshold `p_c ≈ 0.6447` bounds the whole phenomenon. Below it
  altruism is both valuable and personally expensive; just above it the selfish
  equilibrium is already optimal, and altruism becomes simultaneously useless and
  free.
* Adding altruists is not even monotonically helpful: on ~9% of networks there is
  a range of `α` over which converting selfish drivers into altruists makes the
  average commute *worse*. The effect is real but small (≤0.4% of `C_opt`).

## Contents

```
poi/lattice.py   the paper's directed lattice, plus Pigou and a 3D BCC variant
poi/qp.py        the mixed-equilibrium convex QP (Clarabel) + a degeneracy certificate
poi/pigou.py     exact closed-form solution of the two-road case
poi/metrics.py   efficiency, saturation, exploitation gap, backfire
poi/sweep.py     parallel disorder-averaged sweeps
experiments/     compute.py builds results/*.npz; plots.py renders figures/*.png
tests/           222 tests: paper sanity checks, closed form vs solver, equilibrium conditions
```

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # ~5 s
python experiments/compute.py       # ~40 min on 4 cores
python experiments/plots.py
```

## The model

The lattice is the paper's: a square lattice rotated 45°, unidirectional to the
right, periodic vertically, `L` vertices per layer and `2L` layers, so there are
`4L²` roads and every path crosses exactly `2L` of them. Each road is
**congestible** with probability `p`, costing `c(x) = x`, and otherwise
**constant**, costing `c(x) = 1`. One unit of traffic crosses from the left
boundary to the right.

Two checks fix the conventions, and both are reproduced exactly by the solver: at
`p = 0` every road costs 1 so `C = 2L`; at `p = 1` symmetry forces `x = 1/(2L)`
everywhere so `C = 1`.

## Why this is a QP at all

This is the part that is not obvious. Write `f^A`, `f^S` for the two classes'
flows and `x = f^A + f^S`. A selfish driver responds to the travel time she
experiences, `c_e(x) = a_e + b_e x`; an altruist responds to the marginal social
cost `d/dx[x c_e(x)] = a_e + 2 b_e x`. Taken at face value that heterogeneous
equilibrium has the Jacobian

```
[ 2b  2b ]
[  b   b ]
```

which is not symmetric, and whose symmetric part is **indefinite** — so there is
no potential function, no convex program, and in general no uniqueness.

But an altruist only ever *compares* her own perceived path costs, so rescaling
her perceived cost by any `λ > 0` leaves the equilibrium unchanged. Rescaling
gives the symmetric-part condition `8λ ≥ (2λ + 1)²`, i.e. `(2λ - 1)² ≤ 0`. There
is **exactly one** rescaling that works, `λ = 1/2`, at which the Jacobian becomes
the symmetric PSD matrix `[[b, b], [b, b]]` and a potential exists:

```
Φ(f^A, f^S) = Σ_e [ (a_e/2)·f^A_e + a_e·f^S_e + (b_e/2)·(f^A_e + f^S_e)² ]
```

Minimising `Φ` over the two-commodity flow polytope *is* the mixed equilibrium,
and it is a convex QP. It also interpolates the paper exactly: at `α = 0` it is
Beckmann's potential (the selfish equilibrium), and at `α = 1` it is `C/2`, whose
minimiser is the social optimum. So `α = 0` and `α = 1` reproduce Skinner's two
problems, and the price of anarchy is just `C(0)/C(1)`.

`Φ` is only positive *semi*-definite, so the minimiser need not be unique. A tiny
ridge selects the minimum-norm one; `poi.qp.cost_range` then certifies the true
spread by solving two LPs over the exact solution set (the cost is *linear*
there, because the gradient is constant across the set and therefore pins `x` on
every congestible road). Measured spread is `~3e-7` — the reported costs are
unique to seven digits.

## Results

### 1. Pigou's two roads, exactly

![Pigou](figures/fig1_pigou_exact.png)

With a constant road `c₁ = 1` beside a congestible one `c₂ = x`, the mixed
equilibrium is piecewise closed-form. Selfish drivers overfill road 2; altruists
retreat to road 1 until road 2's usage drops to its optimal `1/2`:

| regime | | |
|---|---|---|
| `α ≤ 1/2` | `x₂ = 1 - α` | `C = 1 - α + α²` |
| `α ≥ 1/2` | `x₂ = 1/2` | `C = 3/4` (the social optimum) |

Three things fall straight out, and all three survive on the lattice:

* **Saturation at `α* = 1/2`.** Half the population being altruistic captures
  *the entire* achievable gain. Below that, the share captured is exactly
  `4α(1-α)` — the first altruists are worth far more than the last.
* **The altruists get none of it.** For all `α ≤ 1/2` an altruist experiences
  `C_A = 1`, exactly the full-anarchy commute time, while a selfish driver
  experiences `C_S = 1 - α`, improving all the way to `1/2`. The entire benefit
  of cooperation is transferred to the people who did not cooperate.
* **Exploitation peaks exactly where the gains run out.** The gap `C_A - C_S`
  rises to `1/2` at `α = 1/2` and then decays as `1/(4α)`.

Above `α*` altruists do finally start to benefit (`C_A = 1/2 + 1/(4α)`), but they
never catch the selfish, who sit at `1/2` throughout.

### 2. The paper, reproduced

![POA](figures/fig2_paper_poa.png)
![Scaling](figures/fig3_paper_scaling.png)

Setting `α ∈ {0, 1}` recovers Skinner's results: the price of anarchy peaks at
the directed-percolation threshold `p_c ≈ 0.6447`, sharpens with `L`, and the
peak location converges to `p_c`. The commute time scales as `C ∝ L¹` below
`p_c`, `C ∝ L⁰` above it, and as a nontrivial power at threshold.

### 3. Does altruism speed things up?

![Altruism curves](figures/fig4_altruism_curves.png)

Yes — and the Pigou structure survives the disorder. Total commute time falls
with `α`, but the curve is strongly concave: the first altruists do most of the
work. At `p_c`, a quarter of the population being altruistic already captures
**53%** of the achievable gain and half captures **86%**.

The distributional picture is worse than Pigou's, though. At `p_c` the altruists'
own commute *starts above* the pure-anarchy cost and stays there across **69%**
of the `α` range — peaking at 9.2% worse than `C_eq`. On the lattice altruists
are strictly worse off than *anyone* was before altruism was introduced, not
merely no better off as in Pigou. Meanwhile the selfish drivers end up **14%
faster than the social optimum's average** — they are not just free-riding on the
gain, they are consuming more than all of it.

| at `p = p_c`, `L = 20` | value |
|---|---|
| price of anarchy `C(0)/C(1)` | 1.051 |
| gain captured at `α = 0.25` | 53% |
| gain captured at `α = 0.50` | 86% |
| mean saturating fraction `α*` | 0.64 |
| peak `C_A - C_S` | 0.66 (17% of `C_opt`) |
| altruists worse off than full anarchy | over 69% of the `α` range |
| best selfish commute vs `C_opt` | 14% faster |

### 4. What the percolation threshold actually controls

![Heatmaps](figures/fig5_heatmaps.png)
![Saturation](figures/fig6_saturation.png)

Skinner's result is that *inefficiency* is maximal at `p_c`, and that reproduces.
The distributional quantities behave differently, and it is worth being precise
about how — the naive guess that they also peak at `p_c` is wrong:

* The **absolute** altruist penalty `C_A - C_S` peaks near `p ≈ 0.23`, far below
  `p_c`. That is not a real effect: below `p_c` traffic must cross many constant
  roads, so `C ∝ L` and *every* time difference is larger down there.
* Normalised by the cost scale, `(C_A - C_S)/C_opt` instead rises to a **plateau
  of ≈ 22%** spanning roughly `0.55 < p < 0.78` — straddling `p_c` — and then
  collapses to zero within a few percent of `p_c`.
* The saturating fraction `α*` behaves the same way: ≈ 0.85 below `p_c`,
  falling through zero just above it.

So `p_c` is the **upper edge** of the regime in which altruism means anything,
not a peak in it. Above `p_c` there are many parallel all-congestible routes, the
selfish equilibrium is already optimal (`POA → 1`), and altruism is at once
worthless and costless. Below `p_c` there are gains to be had, and someone has to
pay for them.

### 5. Altruism can backfire

![Backfire](figures/fig7_backfire.png)

Because the mixed equilibrium minimises a potential rather than the total cost,
`C(α)` need not decrease. On a minority of random networks, converting selfish
drivers into altruists makes the *average* commute worse over a range of `α`.
This is a genuine effect, not solver noise: a certified example at `p = 0.4`,
`L = 10` dips to `C = 6.9965` at `α = 0.20` and then climbs to `C = 7.0286` at
`α = 0.28`, with all costs certified unique to 7 digits by `cost_range`.

It is also small, and worth not overselling: across 2976 random networks, 8.8%
show some backfiring range, the largest rise on any single network is 0.38% of
`C_opt`, and only 0.16% of all `(network, α)` pairs are actually worse than full
anarchy. Like everything else here, it vanishes above `p_c`.

### 6. Where each class drives

![Traffic](figures/fig8_traffic_maps.png)

The intuitive story — "altruists take the slow roads" — turns out to be wrong,
and the data says so: at `p ≈ p_c` **both** classes do ~98% of their travel on
congestible roads. What actually differs is *concentration*. At `α = 0.5` the
selfish squeeze onto 1241 roads with their busiest tenth carrying 62% of their
travel; the altruists spread the same amount of traffic over 1552 roads, with
only 47% on their busiest tenth. Altruists do not avoid the fast roads — they
decline to pile onto the *same* fast roads, which is what leaves those roads
clear for the selfish. This matches the paper's own Fig. 10, where the optimum
has a visibly more even flow distribution than the equilibrium.

## Caveats

* Costs are affine (`c = 1` or `c = x`), matching the paper. Roughgarden–Tardos
  bounds the price of anarchy by `4/3` here; nonlinear or jamming costs would
  make every effect larger.
* "Altruistic" means *myopically* altruistic — routing on marginal social cost,
  taking others' choices as given. A central planner who anticipated the selfish
  response (Stackelberg routing) would do better, and is a bilevel problem, not
  a QP.
* The boundary condition is the paper's circuit analogy: left and right
  boundaries are equipotential busbars. `entry="uniform"` forces `1/L` in at each
  left vertex instead; results are qualitatively identical (tested).
* Interior-point solutions pinned to a vertex carry `~1e-6` absolute error. Every
  effect reported here is `1e-2` or larger.
