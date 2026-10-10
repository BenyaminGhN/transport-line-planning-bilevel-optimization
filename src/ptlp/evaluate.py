"""x -> feasibility -> y*(x) -> objective + metrics. Shared by exact enumeration and GA."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assignment import AssignmentResult, PassengerAssignment
from .instance import Instance


@dataclass
class EvaluationResult:
    x: np.ndarray
    feasible: bool
    reason: str                 # 'ok' | 'budget' | 'fleet' | 'demand_not_servable'
    c_op: float
    c_pass: float
    objective: float            # lambda_op*c_op/ref + lambda_pass*c_pass/ref (np.inf if infeasible)
    gini: float | None
    p95: float | None
    assignment: AssignmentResult | None


class Evaluator:
    def __init__(self, instance: Instance, capacitated: bool = True,
                 use_cache: bool = True, penalty: float = 1e6):
        raise NotImplementedError

    def evaluate(self, x: np.ndarray) -> EvaluationResult:
        """Input: 0/1 vector. Output: EvaluationResult. Cached by x.tobytes()."""
        raise NotImplementedError

    def check_plan(self, x: np.ndarray) -> tuple[bool, str]:
        """Budget (2) and fleet (3) checks only (cheap, no LP). Output: (ok, reason)."""
        raise NotImplementedError

    def fitness(self, result: EvaluationResult) -> float:
        """Output: value the GA minimises (objective, or penalised value if infeasible)."""
        raise NotImplementedError

    @property
    def n_lp_solves(self) -> int: raise NotImplementedError


def compute_reference_values(instance: Instance) -> tuple[float, float]:
    """Output: (C_op_ref, C_pass_ref) from the all-lines-active plan (uncapacitated if capacity makes it infeasible)."""
    raise NotImplementedError
