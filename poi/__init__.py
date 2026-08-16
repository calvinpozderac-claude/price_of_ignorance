"""Mixed altruistic/selfish traffic equilibria on random congestible networks.

Built on Skinner, *The price of anarchy is maximized at the percolation
threshold*, arXiv:1404.2935.
"""

from .lattice import (
    Network,
    bcc_lattice,
    pigou_network,
    square_lattice,
    uniform_entry_demand,
)
from .pigou import PigouSolution, pigou_curves, solve_pigou
from .qp import Solution, cost_range, solve_mixed, solve_single_class

__all__ = [
    "Network",
    "square_lattice",
    "bcc_lattice",
    "pigou_network",
    "uniform_entry_demand",
    "Solution",
    "solve_mixed",
    "solve_single_class",
    "cost_range",
    "PigouSolution",
    "solve_pigou",
    "pigou_curves",
]

__version__ = "0.1.0"

PC_SQUARE = 0.6447  # directed bond percolation threshold, square lattice
PC_BCC = 0.2873
