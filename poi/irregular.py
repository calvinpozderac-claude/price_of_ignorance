"""Networks that deliberately break the lattice's special structure.

Both source papers use a lattice in which *every* path crosses exactly ``2L``
roads.  That is not a cosmetic detail: it is what makes ``sum_e x_e = 2L`` hold
for every feasible flow, which in turn is the "free direction" that the
:func:`poi.ignorance.gamma_star` derivation leans on, and it removes any
possibility of ignorance sending traffic down a *longer* route.

Real road networks have nothing of the kind.  The generators here keep
everything else about the model identical -- the same two road types, the same
``c = 1`` and ``c = x`` cost pair, the same single source-to-sink unit demand --
and change only the path-length structure, so that any difference in the results
is attributable to that one assumption.

``skip_dag``
    A layered DAG with extra links that jump two layers ahead.  Paths then cross
    anywhere between roughly ``K/2`` and ``K`` roads.
"""

from __future__ import annotations

import numpy as np

from .lattice import Network, _build

__all__ = ["skip_dag", "path_length_spread"]


def skip_dag(
    K: int = 24,
    W: int = 12,
    p: float = 0.5,
    rng=None,
    skip_frac: float = 0.5,
) -> Network:
    """Layered DAG with skip links, so paths cross differing numbers of roads.

    Parameters
    ----------
    K:
        Number of layers of roads (the lattice analogue is ``K = 2L``).
    W:
        Width -- nodes per layer, periodic in the transverse direction.
    p:
        Probability that a road is congestible (``c = x``); otherwise ``c = 1``.
    skip_frac:
        Probability that each node also gets a link jumping two layers ahead.
        ``skip_frac = 0`` recovers a plain layered lattice with equal path
        lengths; larger values widen the spread of path lengths.
    """
    if not 0.0 <= skip_frac <= 1.0:
        raise ValueError("skip_frac must lie in [0, 1]")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    rng = np.random.default_rng(rng)

    def nid(t, y):
        return t * W + (y % W)

    n_grid = (K + 1) * W
    edges = []
    for t in range(K):
        for y in range(W):
            for dy in (0, 1):
                edges.append((nid(t, y), nid(t + 1, y + dy)))
            if t + 2 <= K and rng.random() < skip_frac:
                edges.append((nid(t, y), nid(t + 2, y)))

    m = len(edges)
    congestible = rng.random(m) < p
    a = np.where(congestible, 0.0, 1.0)
    b = np.where(congestible, 1.0, 0.0)

    src, snk = n_grid, n_grid + 1
    for y in range(W):
        edges.append((src, nid(0, y)))
    for y in range(W):
        edges.append((nid(K, y), snk))
    a = np.concatenate([a, np.zeros(2 * W)])
    b = np.concatenate([b, np.zeros(2 * W)])

    net = _build(edges, n_grid + 2, a, b, src, snk)
    net.meta = {
        "kind": "skip_dag",
        "K": K,
        "W": W,
        "p": p,
        "skip_frac": skip_frac,
        "n_lattice_edges": m,
        "congestible_mask": np.concatenate([congestible, np.zeros(2 * W, dtype=bool)]),
    }
    return net


def path_length_spread(net: Network) -> tuple[int, int]:
    """Shortest and longest number of *real* roads on any source-to-sink path.

    Equal values mean the network has the lattice's special structure, so
    ``sum_e x_e`` is the same for every feasible flow.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra

    real = net.is_real_road
    n = net.n_nodes
    lo = np.where(real, 1.0, 0.0)
    g_min = sp.csr_matrix((lo + 1e-9, (net.tail, net.head)), shape=(n, n))
    shortest = dijkstra(g_min, indices=net.source, min_only=True)[net.sink]

    # Longest path on a DAG: negate and relax in topological order.
    order = np.argsort(net.tail, kind="stable")
    best = np.full(n, -np.inf)
    best[net.source] = 0.0
    for _ in range(2):  # source has the highest index, so relay it first
        for e in order:
            u, v = net.tail[e], net.head[e]
            if best[u] > -np.inf:
                best[v] = max(best[v], best[u] + (1.0 if real[e] else 0.0))
    return int(round(shortest)), int(best[net.sink])
