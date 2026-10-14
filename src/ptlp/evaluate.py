"""x -> feasibility -> y*(x) -> objective + metrics. Shared by exact enumeration and GA."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assignment import AssignmentResult, PassengerAssignment
from .instance import Instance
from .metrics import passenger_travel_times, weighted_gini, weighted_percentile


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
        instance.build()
        self.instance = instance
        self.use_cache = use_cache
        self.penalty = float(penalty)
        self.frequencies = np.asarray([line.frequency for line in instance.lines], dtype=float)
        self.assignment = PassengerAssignment(instance, unmet_penalty=penalty,
                                              capacitated=capacitated)
        self._cache: dict[bytes, EvaluationResult] = {}
        self._n_lp_solves = 0
        if instance.c_op_ref is None or instance.c_pass_ref is None:
            c_op_ref, c_pass_ref = compute_reference_values(instance, penalty)
            instance.set_references(c_op_ref, c_pass_ref)

    def evaluate(self, x: np.ndarray) -> EvaluationResult:
        """Input: 0/1 vector. Output: EvaluationResult. Cached by x.tobytes()."""
        x = self._validate_x(x)
        key = x.tobytes()
        if self.use_cache and key in self._cache:
            return self._cache[key]

        c_op = float(self.instance.line_cost @ x)
        plan_ok, reason = self.check_plan(x)
        if not plan_ok:
            result = EvaluationResult(x.copy(), False, reason, c_op, np.inf, np.inf,
                                      None, None, None)
        else:
            assignment = self.assignment.solve(x, self.frequencies)
            self._n_lp_solves += 1
            if not assignment.feasible:
                result = EvaluationResult(x.copy(), False, "assignment_failed", c_op,
                                          np.inf, np.inf, None, None, assignment)
            else:
                passenger_cost = assignment.objective
                objective = (
                    self.instance.lambda_op * c_op / self.instance.c_op_ref
                    + self.instance.lambda_pass * passenger_cost / self.instance.c_pass_ref
                )
                tt, weights = passenger_travel_times(self.instance, assignment.flows)
                gini = weighted_gini(tt, weights)
                p95 = weighted_percentile(tt, weights)
                result = EvaluationResult(x.copy(), True, "ok", c_op, passenger_cost,
                                          float(objective), gini, p95, assignment)
        if self.use_cache:
            self._cache[key] = result
        return result

    def check_plan(self, x: np.ndarray) -> tuple[bool, str]:
        """Budget (2) and fleet (3) checks only (cheap, no LP). Output: (ok, reason)."""
        x = self._validate_x(x)
        if float(self.instance.line_cost @ x) > self.instance.budget + 1e-9:
            return False, "budget"
        if float(self.instance.line_fleet @ x) > self.instance.fleet_total + 1e-9:
            return False, "fleet"
        return True, "ok"

    def fitness(self, result: EvaluationResult) -> float:
        """Output: value the GA minimises (objective, or penalised value if infeasible)."""
        if result.feasible:
            return result.objective
        return float(self.penalty)

    @property
    def n_lp_solves(self) -> int:
        return self._n_lp_solves

    def _validate_x(self, x: np.ndarray) -> np.ndarray:
        try:
            x = np.asarray(x, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("x must be a binary vector with one entry per line") from exc
        if (x.shape != (self.instance.n_lines,)
                or np.any(~np.isfinite(x))
                or not np.all((x == 0) | (x == 1))):
            raise ValueError("x must be a binary vector with one entry per line")
        return x.astype(np.uint8, copy=False)


def compute_reference_values(instance: Instance, unmet_penalty: float = 1e6) -> tuple[float, float]:
    """Output: (C_op_ref, C_pass_ref) from the all-lines-active plan (uncapacitated if capacity makes it infeasible)."""
    instance.build()
    x = np.ones(instance.n_lines, dtype=np.uint8)
    frequencies = np.asarray([line.frequency for line in instance.lines], dtype=float)
    assignment = PassengerAssignment(instance, unmet_penalty=unmet_penalty,
                                     capacitated=False).solve(x, frequencies)
    if not assignment.feasible:
        raise RuntimeError(f"reference assignment failed: {assignment.status}")
    # A unit fallback keeps normalization defined for zero-cost toy instances.
    return max(float(instance.line_cost.sum()), 1.0), max(float(assignment.objective), 1.0)
