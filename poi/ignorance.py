"""User ignorance about road types, following Saray, Pozderac, Josephson & Skinner.

*The benefit of ignorance for traffic through a random congestible network*,
arXiv:2503.09684.

The idea
--------
Drivers do not reliably know which roads are the fast, congestible ones.  Each
driver plans against a *perceived* commute time that mixes the road's true cost
function with the cost function of the opposite road type, weighted by an
ignorance level.  With ``u = omega / 2``:

    slow road (true ``c = 1``):  ``c^p(x) = (1 - u) + u * x``
    fast road (true ``c = x``):  ``c^p(x) = u + (1 - u) * x``

``omega = 0`` is perfect knowledge; ``omega = 1`` is total ignorance, where every
road looks identical (``c^p = (1 + x) / 2``) and the traffic spreads uniformly.

Notation
--------
The ignorance paper calls this parameter ``alpha``.  This repository already
uses ``alpha`` for the *altruistic fraction* from the price-of-anarchy work, so
ignorance is written ``omega`` throughout.  The mapping is simply

    omega (here)  ==  alpha (arXiv:2503.09684)

Why this composes with the altruism model for free
--------------------------------------------------
Perceived costs stay affine, ``c^p_e(x) = a^p_e + b^p_e x``, and
:mod:`poi.qp` already solves the mixed altruistic/selfish equilibrium for
*arbitrary* affine costs.  So ignorance needs no new solver -- only a
transformation of the cost coefficients:

* a **selfish** driver responds to the perceived cost ``a^p + b^p x``;
* an **altruistic** driver responds to the perceived *marginal social* cost
  ``d/dx [x c^p(x)] = a^p + 2 b^p x``.

The second point is the substantive modelling choice: an ignorant altruist tries
to minimise the total commute time *as she believes it to be*.  She therefore
aims at the wrong target whenever ``omega > 0``, which is what makes altruism and
ignorance interact rather than simply add.

Outcomes are always scored with the **true** cost functions.  Pass the perceived
network to ``solve_mixed`` as ``net`` and the true one as ``true_net``.
"""

from __future__ import annotations

import numpy as np

from .lattice import Network

__all__ = [
    "perceive",
    "solve_ignorant",
    "OMEGA_STAR",
    "objective_identity_residual",
]

# Ignorance level at which the purely selfish equilibrium reproduces the social
# optimum in the large-system limit (arXiv:2503.09684, Sec. IV).  It arises as
# the point where beta = 3*omega/2 - 1 changes sign in the paper's proof.
OMEGA_STAR = 2.0 / 3.0


def perceive(net: Network, omega: float) -> Network:
    """Return a copy of ``net`` carrying the perceived costs at ignorance ``omega``.

    Roads are classified by their true cost function: a road with ``b_e > 0`` is
    congestible ("fast"), one with ``b_e == 0`` and ``a_e > 0`` is constant
    ("slow").  Zero-cost connector edges (the super-source/-sink links) are left
    untouched, since there is nothing to be ignorant about.

    The construction assumes the paper's canonical cost pair ``c = 1`` and
    ``c = x``; a network built with other coefficients is rescaled to that pair
    before mixing, so ``perceive`` is only meaningful for the standard model.
    """
    if not 0.0 <= omega <= 1.0:
        raise ValueError("omega must lie in [0, 1]")
    u = 0.5 * omega

    fast = net.b > 0
    slow = (~fast) & (net.a > 0)

    a = np.zeros_like(net.a)
    b = np.zeros_like(net.b)
    # slow: (1 - u) + u x        fast: u + (1 - u) x
    a[slow] = 1.0 - u
    b[slow] = u
    a[fast] = u
    b[fast] = 1.0 - u

    out = Network(
        a=a,
        b=b,
        tail=net.tail,
        head=net.head,
        n_nodes=net.n_nodes,
        source=net.source,
        sink=net.sink,
        incidence=net.incidence,
        pos=net.pos,
        meta={**(net.meta or {}), "omega": float(omega), "perceived": True},
    )
    return out


def solve_ignorant(net: Network, omega: float, alpha: float, **kw):
    """Mixed equilibrium of ignorant drivers; costs scored with the true network.

    ``net`` carries the true cost functions.  Drivers respond to
    ``perceive(net, omega)``, a fraction ``alpha`` of them as altruists.
    """
    from .qp import solve_mixed

    return solve_mixed(perceive(net, omega), alpha, true_net=net, **kw)


def objective_identity_residual(net: Network, x: np.ndarray) -> float:
    """Check the paper's exact identity at ``omega = 2/3`` (their Eq. 12).

    ``F_{2/3}(x) = (1/3) C(x) + (1/6) sum_slow x^2 + (2/3) L``, using
    ``sum_all x = 2L``.  Returns the absolute residual, which should be ~0 for
    any feasible flow (it is an algebraic identity, not a property of the
    optimum).
    """
    fast = net.b > 0
    slow = (~fast) & (net.a > 0)
    real = fast | slow

    u = 1.0 / 3.0  # omega / 2 at omega = 2/3
    F = np.sum(u * x[slow] ** 2 / 2 + (1 - u) * x[slow]) + np.sum(
        (1 - u) * x[fast] ** 2 / 2 + u * x[fast]
    )
    C = np.sum(x[slow]) + np.sum(x[fast] ** 2)
    rhs = C / 3.0 + np.sum(x[slow] ** 2) / 6.0 + np.sum(x[real]) / 3.0
    return float(abs(F - rhs))
