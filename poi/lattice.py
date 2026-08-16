"""Directed lattice networks with congestible and constant roads.

This reproduces the network of Skinner, "The price of anarchy is maximized at the
percolation threshold" (arXiv:1404.2935), Fig. 1(b): a square lattice rotated 45
degrees, unidirectional in the +x direction, periodic in the vertical direction.

Geometry (system size L)
------------------------
Nodes are ``(t, y)`` for ``t = 0 .. 2L`` (layer index, flow moves to larger t)
and ``y = 0 .. L-1`` (vertical position, periodic).  Every node in layer
``t < 2L`` has exactly two outgoing roads::

    (t, y) -> (t+1, y)            "down"
    (t, y) -> (t+1, (y+1) % L)    "up"

This gives

* ``L`` vertices on the left boundary and ``L`` on the right boundary,
* path length exactly ``2L`` roads,
* ``2 ** (2L)`` distinct paths from any left vertex,
* ``4 L**2`` roads in total,

all four of which match the counts quoted in the paper.  The paper's sanity
checks follow: at ``p = 1`` symmetry forces ``x = 1/(2L)`` on every road so
``C = 4L^2 * (1/2L)^2 = 1``; at ``p = 0`` every road costs 1 and ``C = 2L``.

Boundary conditions
-------------------
A super-source ``S`` feeds all ``L`` left vertices and all ``L`` right vertices
drain into a super-sink ``T``, both through zero-cost roads.  This is the
"busbar" reading of the paper's electrical-circuit analogy, where the left and
right boundaries are equipotential electrodes carrying a unit total current.
``entry="uniform"`` instead pins exactly ``1/L`` of the traffic into each left
vertex; results are qualitatively identical (see ``tests/test_lattice.py``).

Road types
----------
Each lattice road is *congestible* with probability ``p``, giving a cost
``c(x) = x`` (``a = 0``, ``b = 1``), and otherwise *constant*, ``c(x) = 1``
(``a = 1``, ``b = 0``).  Roads are stored in the general affine form
``c_e(x) = a_e + b_e x`` so other cost families drop in unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

__all__ = ["Network", "square_lattice", "pigou_network", "bcc_lattice"]


@dataclass
class Network:
    """A directed network with affine link costs ``c_e(x) = a_e + b_e x``.

    Attributes
    ----------
    a, b:
        Length-``n_edges`` cost coefficients.
    tail, head:
        Length-``n_edges`` node indices.
    n_nodes:
        Number of nodes; ``source`` and ``sink`` index into that range.
    incidence:
        Sparse ``(n_nodes, n_edges)`` matrix with ``+1`` at the tail and ``-1``
        at the head, so ``incidence @ f`` is the net outflow at each node.
    pos:
        Optional ``(n_nodes, 2)`` array of drawing coordinates.
    meta:
        Free-form dictionary describing how the network was generated.
    """

    a: np.ndarray
    b: np.ndarray
    tail: np.ndarray
    head: np.ndarray
    n_nodes: int
    source: int
    sink: int
    incidence: sp.csc_matrix
    pos: np.ndarray | None = None
    meta: dict | None = None

    @property
    def n_edges(self) -> int:
        return int(self.a.size)

    @property
    def is_congestible(self) -> np.ndarray:
        """Boolean mask of roads whose cost grows with usage."""
        return self.b > 0

    @property
    def is_real_road(self) -> np.ndarray:
        """Boolean mask excluding zero-cost source/sink connector edges."""
        return (self.a > 0) | (self.b > 0)

    def demand(self, total: float = 1.0) -> np.ndarray:
        """Node-balance right-hand side moving ``total`` flow from source to sink."""
        d = np.zeros(self.n_nodes)
        d[self.source] = total
        d[self.sink] = -total
        return d

    def costs(self, x: np.ndarray) -> np.ndarray:
        """Realised travel time on each road at usage ``x``."""
        return self.a + self.b * x

    def marginal_costs(self, x: np.ndarray) -> np.ndarray:
        """Marginal social cost ``d/dx [x c(x)] = a + 2 b x`` on each road."""
        return self.a + 2.0 * self.b * x

    def total_cost(self, x: np.ndarray) -> float:
        """Total (= average, since demand is 1) commute time ``sum_e x_e c_e(x_e)``."""
        return float(np.sum(x * self.costs(x)))


def _build(edges, n_nodes, a, b, source, sink, pos=None, meta=None) -> Network:
    tail = np.asarray([e[0] for e in edges], dtype=np.int64)
    head = np.asarray([e[1] for e in edges], dtype=np.int64)
    m = tail.size
    rows = np.concatenate([tail, head])
    cols = np.concatenate([np.arange(m), np.arange(m)])
    vals = np.concatenate([np.ones(m), -np.ones(m)])
    inc = sp.csc_matrix((vals, (rows, cols)), shape=(n_nodes, m))
    return Network(
        a=np.asarray(a, dtype=float),
        b=np.asarray(b, dtype=float),
        tail=tail,
        head=head,
        n_nodes=n_nodes,
        source=source,
        sink=sink,
        incidence=inc,
        pos=pos,
        meta=meta or {},
    )


def square_lattice(L: int, p: float, rng=None, entry: str = "busbar") -> Network:
    """Build the paper's 2D directed lattice with a fraction ``p`` of congestible roads.

    Parameters
    ----------
    L:
        System size: ``L`` vertices per layer, ``2L`` layers of roads.
    p:
        Probability that any given road is congestible (``c(x) = x``).
    rng:
        ``numpy`` generator or integer seed for the road-type disorder.
    entry:
        ``"busbar"`` (default) attaches a super-source/-sink to the whole
        boundary; ``"uniform"`` forces exactly ``1/L`` in at each left vertex.
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if entry not in ("busbar", "uniform"):
        raise ValueError("entry must be 'busbar' or 'uniform'")
    rng = np.random.default_rng(rng)

    n_layers = 2 * L  # layers of roads
    def nid(t, y):
        return t * L + (y % L)

    n_grid = (n_layers + 1) * L
    edges, a, b, kinds = [], [], [], []
    pos = [(0.0, 0.0)] * n_grid

    for t in range(n_layers + 1):
        for y in range(L):
            # draw the lattice sheared so the two out-edges fan up and down
            pos[nid(t, y)] = (float(t), (y - 0.5 * t) % L)

    n_lattice_edges = 4 * L * L
    congestible = rng.random(n_lattice_edges) < p
    k = 0
    for t in range(n_layers):
        for y in range(L):
            for dy in (0, 1):
                edges.append((nid(t, y), nid(t + 1, y + dy)))
                if congestible[k]:
                    a.append(0.0)
                    b.append(1.0)
                else:
                    a.append(1.0)
                    b.append(0.0)
                kinds.append(bool(congestible[k]))
                k += 1

    src = n_grid
    snk = n_grid + 1
    n_nodes = n_grid + 2
    pos = pos + [(-1.5, (L - 1) / 2.0), (n_layers + 1.5, (L - 1) / 2.0)]

    if entry == "busbar":
        for y in range(L):
            edges.append((src, nid(0, y)))
            a.append(0.0)
            b.append(0.0)
            kinds.append(False)
        for y in range(L):
            edges.append((nid(n_layers, y), snk))
            a.append(0.0)
            b.append(0.0)
            kinds.append(False)
        net = _build(edges, n_nodes, a, b, src, snk, np.asarray(pos))
    else:
        # Uniform injection: no source edges, node demands are set directly.
        for y in range(L):
            edges.append((nid(n_layers, y), snk))
            a.append(0.0)
            b.append(0.0)
            kinds.append(False)
        net = _build(edges, n_nodes, a, b, src, snk, np.asarray(pos))

    net.meta = {
        "kind": "square_lattice",
        "L": L,
        "p": p,
        "entry": entry,
        "n_lattice_edges": n_lattice_edges,
        "congestible_mask": np.asarray(kinds, dtype=bool),
        "path_length": 2 * L,
        "pc": 0.6447,  # directed bond percolation threshold, Jensen (1999)
    }
    if entry == "uniform":
        net.meta["uniform_entry_nodes"] = np.asarray([nid(0, y) for y in range(L)])
    return net


def uniform_entry_demand(net: Network, total: float = 1.0) -> np.ndarray:
    """Demand vector for ``entry="uniform"``: ``total/L`` injected at each left vertex."""
    nodes = net.meta["uniform_entry_nodes"]
    d = np.zeros(net.n_nodes)
    d[nodes] = total / len(nodes)
    d[net.sink] = -total
    return d


def pigou_network(a2: float = 0.0, b2: float = 1.0, c1: float = 1.0) -> Network:
    """Pigou's two-road example: a constant road ``c1`` beside a congestible one.

    Defaults reproduce the paper's Fig. 1(a): ``c_1(x) = 1``, ``c_2(x) = x``.
    """
    edges = [(0, 1), (0, 1)]
    net = _build(
        edges,
        n_nodes=2,
        a=[c1, a2],
        b=[0.0, b2],
        source=0,
        sink=1,
        pos=np.asarray([(0.0, 0.0), (1.0, 0.0)]),
        meta=None,
    )
    net.meta = {"kind": "pigou", "c1": c1, "a2": a2, "b2": b2}
    return net


def bcc_lattice(L: int, p: float, rng=None) -> Network:
    """3D body-centred-cubic analogue of :func:`square_lattice` (paper Appendix A).

    Each node has four outgoing roads, one to each diagonal neighbour in the
    next layer, lowering the directed percolation threshold to ``pc ~ 0.2873``.
    Layers run ``t = 0 .. 2L`` with an ``L x L`` periodic cross-section.
    """
    rng = np.random.default_rng(rng)
    n_layers = 2 * L

    def nid(t, y, z):
        return t * L * L + (y % L) * L + (z % L)

    n_grid = (n_layers + 1) * L * L
    edges, a, b, kinds = [], [], [], []
    n_lattice_edges = 4 * n_layers * L * L
    congestible = rng.random(n_lattice_edges) < p
    k = 0
    for t in range(n_layers):
        for y in range(L):
            for z in range(L):
                for dy in (0, 1):
                    for dz in (0, 1):
                        edges.append((nid(t, y, z), nid(t + 1, y + dy, z + dz)))
                        if congestible[k]:
                            a.append(0.0)
                            b.append(1.0)
                        else:
                            a.append(1.0)
                            b.append(0.0)
                        kinds.append(bool(congestible[k]))
                        k += 1

    src, snk = n_grid, n_grid + 1
    for y in range(L):
        for z in range(L):
            edges.append((src, nid(0, y, z)))
            a.append(0.0)
            b.append(0.0)
            kinds.append(False)
    for y in range(L):
        for z in range(L):
            edges.append((nid(n_layers, y, z), snk))
            a.append(0.0)
            b.append(0.0)
            kinds.append(False)

    net = _build(edges, n_grid + 2, a, b, src, snk)
    net.meta = {
        "kind": "bcc_lattice",
        "L": L,
        "p": p,
        "n_lattice_edges": n_lattice_edges,
        "congestible_mask": np.asarray(kinds, dtype=bool),
        "path_length": n_layers,
        "pc": 0.2873,
    }
    return net
