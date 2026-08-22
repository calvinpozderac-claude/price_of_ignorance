# The model: every variable, and how a traffic pattern is produced

This is the reference for what each symbol means and the exact route from a set
of parameters to a flow vector. Everything here is implemented in `poi/`; the
worked trace at the end is real output, not a sketch.

---

## 1. Variables

### 1.1 Network primitives (the road layout)

| symbol | code | meaning |
|---|---|---|
| `e` | edge index | a directed road |
| `m` | `net.n_edges` | number of roads |
| `A` | `net.incidence` | node × edge incidence, `+1` at tail, `-1` at head |
| `a_e` | `net.a` | constant part of the cost of road `e` |
| `b_e` | `net.b` | congestion slope of road `e` |
| `c_e(x)` | `net.costs` | travel time on road `e`: **`a_e + b_e·x_e`** (affine) |

On the papers' lattice each road is one of exactly two kinds:

* **congestible / "fast"**: `(a, b) = (0, 1)`, so `c(x) = x` — free when empty,
  slow when busy.
* **constant / "slow"**: `(a, b) = (1, 0)`, so `c(x) = 1` — never congests.

Zero-cost connector edges `(0, 0)` attach a super-source and super-sink to the
boundary; they are excluded from every road-level statistic by
`net.is_real_road`.

For **real networks** (`poi/bpr.py`) the cost is the Bureau of Public Roads form
instead, and `(a, b)` are replaced by:

| symbol | code | meaning |
|---|---|---|
| `t⁰_e` | `net.t0` | free-flow travel time (minutes) |
| `cap_e` | `net.capacity` | capacity (veh/h) |
| `b_e` | `net.b` | BPR coefficient, 0.15 in the benchmark data |
| `β` | `net.power` | BPR exponent, 4 — **must be the same on every link** |

giving `c_e(x) = t⁰_e (1 + b_e (x/cap_e)^β)`.

### 1.2 Lattice construction parameters

| symbol | code | meaning |
|---|---|---|
| `L` | `L` | system size: `L` nodes per layer, `2L` layers, `4L²` roads |
| `p` | `p` | probability a road is congestible |
| `p_c` | `PC_SQUARE` | directed bond percolation threshold, ≈ 0.6447 |

Because every source→sink path crosses exactly `2L` roads, **`Σ_e x_e = 2L` for
every feasible flow**. That invariant is what the `γ*` derivation uses, and
`poi/irregular.py` exists to break it deliberately.

### 1.3 Demand

| symbol | code | meaning |
|---|---|---|
| `d` | `net.demand()` | node-balance vector; `+1` at source, `−1` at sink |
| `q_od` | `net.demand` (BPR) | origin→destination trip table |

The lattice sends one unit from a boundary-wide source to a boundary-wide sink;
real networks use the benchmark OD matrix (360,600 trips for Sioux Falls).

### 1.4 Behaviour

| symbol | code | meaning |
|---|---|---|
| `α` | `alpha` | **fraction** of drivers who are fully altruistic |
| `γ` | `gamma` | **uniform** altruism level: everyone responds to `c + γ·x·c'` |
| `λ` | — | rescaling of the altruist's perceived cost, `1/(β+1)` |
| `disc_e` | `net.altruist_discount` | `t⁰_e·β/(β+1)`; for affine costs, `a_e/2` |

Two different populations, deliberately not the same thing:

* **selfish** driver minimises her own travel time → responds to `c_e(x)`.
* **altruistic** driver minimises the *total* commute time → responds to the
  marginal social cost `d/dx[x·c(x)] = c + x·c'`.

An altruist only ever *compares* her own path costs, so multiplying her criterion
by any `λ > 0` changes nothing. `λ = 1/(β+1)` is the unique choice that makes the
two classes' Jacobian symmetric, which is what buys a potential function.

### 1.5 Uncertainty

| symbol | code | meaning |
|---|---|---|
| `ω` | `omega` | **systematic** ignorance, 0 = perfect knowledge, 1 = total |
| `u` | — | `ω/2`, the weight put on the wrong road type |
| `σ` | `sigma` | magnitude of **idiosyncratic** perception error |
| `ρ` | `rho` | correlation of that error across drivers |
| `K` | `K` | number of discrete driver types used to represent the error |
| `θ` | `theta` | logit dispersion in `dial_logit`; `θ→∞` is perfect knowledge |
| `ε^k_e` | `eps` | the error type `k` makes on road `e` |

Note the collision hazard: **both source papers call the ignorance level `α`.**
Here `α` is always the altruistic fraction and `ω` is always ignorance.

### 1.6 Outputs

| symbol | code | meaning |
|---|---|---|
| `x_e` | `sol.x` | total traffic on road `e` |
| `f^k_e` | `f_altruist`, `f_selfish` | traffic from class `k` on road `e` |
| `C` | `total_cost` | average commute time, `Σ_e x_e·c_e(x_e)` — scored with **true** costs |
| `C_A`, `C_S` | `cost_altruist`, `cost_selfish` | commute time actually *experienced* by each class |
| `P_A` | — | price of anarchy, `C(α=0) / C(α=1)` |
| `P_I` | — | price of ignorance, `C(ω) / C(ω=0)` |

Two identities always hold and are asserted in the tests:
`C = α·C_A + (1−α)·C_S`, and `C_A ≥ C_S` (the selfish sit on the minimum-latency
paths by definition, so no other class can average below them).

---

## 2. How the traffic pattern is determined

Six steps. The whole thing is one convex program; there is no iteration over
"drivers" and no simulation of individuals.

### Step 1 — build the true network

Draw the road types (`p`), fix the topology (`L`), attach the demand. This gives
`(a_e, b_e)` and `A`, `d`. These are the **true** costs and are used *only* to
score the outcome.

### Step 2 — build the perceived network (systematic bias `ω`)

Drivers plan against a blend of each road's true cost and the opposite type's,
with `u = ω/2`:

```
slow road (true c = 1):  c^p(x) = (1 − u) + u·x
fast road (true c = x):  c^p(x) = u + (1 − u)·x
```

`ω = 0` leaves the network alone; `ω = 1` makes every road look identical,
`c^p = (1+x)/2`. On a real network there are no two types to blend between, so
the analogue shrinks each link's capacity toward the network geometric mean:
`cap̂ = cap^(1−ω) · ḡ^ω`.

### Step 3 — draw the idiosyncratic error (`σ`, `ρ`, `K`)

Split the population into `K` types and give type `k` its own error on each road:

```
ε^k_e = σ·( √ρ·ξ_e  +  √(1−ρ)·η^k_e )
```

`ξ` is drawn once per road and shared by everybody; `η` is drawn per type. The
variance is `σ²` for every `ρ`, so magnitude and correlation move independently.
`ρ = 1` reproduces a shared bias; `ρ = 0` is textbook stochastic user
equilibrium. **`K` is itself a correlation knob** — a handful of types is a
handful of coordinated groups — so it must be pushed until the answer stops
moving (past ~64 on the lattice).

### Step 4 — enumerate the driver classes

Each `(type, disposition)` pair becomes one class with a share and a constant
per-road offset `δ`:

| class | share | offset `δ_e` |
|---|---|---|
| altruist of type `k` | `α/K` | `−disc_e + λ·ε^k_e` |
| selfish of type `k` | `(1−α)/K` | `ε^k_e` |

so there are `2K` classes in general. Every class routes on the perceived cost
`c^p_e(x_e) + δ_e`. The offsets are **constant in the flow**, which is exactly
the property that keeps the problem convex.

### Step 5 — minimise the potential

Because every class's derivative with respect to `x` is the same function
`c'(x)`, the equilibrium conditions are the optimality conditions of a single
convex program:

```
Φ({f^k}) = Σ_e ∫₀^{x_e} c^p_e(s) ds  +  Σ_k Σ_e δ^k_e · f^k_e ,     x = Σ_k f^k
```

minimised over: flow conservation per class (`A f^k = share_k · d`) and
non-negativity (`f^k ≥ 0`).

* **Affine costs** → `Φ` is quadratic → solved exactly by Clarabel
  (`solve_mixed`, `solve_stochastic`). An auxiliary variable `x = Σ_k f^k` keeps
  the Hessian diagonal.
* **BPR costs** → `Φ` is convex but not quadratic → solved by conjugate
  Frank–Wolfe (`frank_wolfe_multiclass`), which alternates shortest-path
  loadings with an exact line search.

The minimiser **is** the equilibrium: at the optimum every class puts flow only
on paths that minimise *that class's* perceived cost.

### Step 6 — score with the true costs

Behaviour follows `c^p`; the answer is always evaluated with `c`:

```
C = Σ_e x_e · c_e(x_e)        C_A = (f^A · c(x)) / α        C_S = (f^S · c(x)) / (1−α)
```

This separation is the whole point of the ignorance work, and it is why
`solve_mixed` takes a `true_net` argument distinct from the network it is
solving on.

---

## 3. A worked trace

Arbitrary parameters: `L = 3`, `p = 0.6`, `ω = 0.4`, `α = 0.3`, `σ = 0.2`,
`ρ = 0.5`, `K = 2`. Real output from `poi`:

```
STEP 1  36 roads + 6 connectors, 23 nodes, path length 6
        first 6 roads (a,b): (0,1) (0,1) (1,0) (0,1) (0,1) (1,0)
        congestible share realised 0.667

STEP 2  u = 0.2:  slow (1,0) -> (0.8, 0.2)    fast (0,1) -> (0.2, 0.8)
        first 6 perceived: (.2,.8) (.2,.8) (.8,.2) (.2,.8) (.2,.8) (.8,.2)

STEP 3  eps shape (2, 42); realised correlation between the two types 0.69
        type 0, first 6: -0.193 -0.111  0.122  0.011 -0.266 -0.019
        type 1, first 6: -0.238  0.118  0.150  0.072 -0.442 -0.210

STEP 4  4 classes = K x {altruist, selfish}
        share 0.15 ALTRUIST  delta[:4] = -0.196 -0.155 -0.339 -0.095
        share 0.35 selfish   delta[:4] = -0.193 -0.111  0.122  0.011
        share 0.15 ALTRUIST  delta[:4] = -0.219 -0.041 -0.325 -0.064
        share 0.35 selfish   delta[:4] = -0.238  0.118  0.150  0.072

STEP 5  x on first 6 roads: 0.406 0.130 0.000 0.091 0.328 0.045
        conservation |A x - d| = 3.6e-12
        sum_e x_e over real roads = 6.0000     (= 2L, as it must be)

STEP 6  true      c_e(x): 0.406 0.130 1.000 0.091 0.328 1.000
        perceived c^p(x): 0.525 0.304 0.800 0.273 0.462 0.809   <- drives choice, never scored

        C   = 2.35302     C_A = 2.75153     C_S = 2.18224
        check  0.3*C_A + 0.7*C_S = 2.35302
```

Two things to read off. Road 2 is a slow road: its true cost is exactly `1.0`
whatever the flow, but drivers perceive it as `0.80` and it ends up carrying zero
traffic. And `C_A > C_S` — the altruists take the longer commute, as they must.

---

## 4. Parameter cheat-sheet

| set this | to get |
|---|---|
| `α = 0, ω = 0, σ = 0` | Wardrop user equilibrium (Skinner's "equilibrium") |
| `α = 1, ω = 0, σ = 0` | system optimum (Skinner's "optimum") |
| `α = 0`, sweep `ω` | the price of ignorance surface (arXiv:2503.09684) |
| sweep `α` at `ω = 0` | the altruism study of Part I |
| `ω = 2/3, α = 0` | ignorant selfish drivers ≈ the social optimum, on the lattice |
| `σ > 0, ρ = 0`, large `K` | stochastic user equilibrium |
| `σ > 0, ρ = 1` | a random shared bias |
| `θ` via `dial_logit` | logit SUE, exact `ρ = 0` continuum, no Monte Carlo |
