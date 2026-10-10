"""Lower level (follower): capacity-constrained min in-vehicle-time LP, solved exactly."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instance import Instance


@dataclass
class AssignmentResult:
    feasible: bool
    status: str                         # HiGHS status text
    flows: np.ndarray | None            # y*, shape (n_routes,)
    c_pass: float                       # sum C_kr * y_kr  (np.inf if infeasible)
    segment_load: np.ndarray | None     # q_ls(y), shape (n_seg,)
    segment_util: np.ndarray | None     # load / capacity, NaN for inactive segments


class PassengerAssignment:
    """Builds the sparse LP structure once; each solve() only changes bounds / RHS."""

    def __init__(self, instance: Instance, capacitated: bool = True, tie_eps: float = 1e-6):
        """capacitated=False gives the uncapacitated baseline (drops constraint (8)).
        tie_eps: deterministic tie-breaking perturbation of route costs."""
        raise NotImplementedError

    def solve(self, x: np.ndarray) -> AssignmentResult:
        """Input: x in {0,1}^|L|. Output: AssignmentResult.
        Steps: _route_upper_bounds(x) -> _capacity_rhs(x) -> linprog(HiGHS)."""
        raise NotImplementedError

    def _route_upper_bounds(self, x: np.ndarray) -> np.ndarray:
        """Constraint (7). Output (n_routes,): D_k if all lines of route active, else 0."""
        raise NotImplementedError

    def _capacity_rhs(self, x: np.ndarray) -> np.ndarray:
        """Constraint (8). Output (n_seg,): seg_capacity * x[seg_line]."""
        raise NotImplementedError

    def _tie_broken_costs(self) -> np.ndarray:
        """Output (n_routes,): route_cost + tie_eps * deterministic rank (document the rule)."""
        raise NotImplementedError
