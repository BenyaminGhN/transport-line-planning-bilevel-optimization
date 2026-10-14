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
    reason: str                 # 'ok' | 'budget' | 'fleet' | 'assignment_failed'
    c_op: float
    c_pass: float               # ride time + omega * unserved   (the follower's objective)
    objective: float            # lambda_op*c_op/ref + lambda_pass*c_pass/ref (np.inf if infeasible)
    gini: float | None          # served passengers only (PDF 9.1)
    p95: float | None           # served passengers only
    assignment: AssignmentResult | None
    # >>> NEW: the numbers the PDF asks to report next to the objective (section 9.2) <<<
    travel_time: float = np.nan      # C_pass without the penalty: total ride time of served passengers
    unserved: float = np.nan         # U(x)
    served_fraction: float = np.nan  # rho_served(x)
    c_op_norm: float = np.nan        # c_op / C_op_ref
    c_pass_norm: float = np.nan      # c_pass / C_pass_ref
    violation: float = 0.0           # how far an infeasible plan is over budget / fleet (0 if feasible)


class Evaluator:
    # >>> CHANGED: the argument 'penalty' is now called 'unmet_penalty' (it is omega of the PDF) and defaults
    # to instance.unmet_penalty, else instance.default_unmet_penalty(). Before, one number (1e6) was used both
    # as omega and as the GA penalty for infeasible plans; omega = 1e6 made the reference value huge and the
    # normalised terms meaningless. They are separate now (see fitness()). <<<
    def __init__(self, instance: Instance, capacitated: bool = True,
                 use_cache: bool = True, unmet_penalty: float | None = None):
        instance.build()
        self.instance = instance
        self.use_cache = use_cache
        if unmet_penalty is None:
            unmet_penalty = instance.unmet_penalty
        if unmet_penalty is None:
            unmet_penalty = instance.default_unmet_penalty()
        self.unmet_penalty = float(unmet_penalty)
        self.frequencies = np.asarray([line.frequency for line in instance.lines], dtype=float)
        self.assignment = PassengerAssignment(instance, unmet_penalty=self.unmet_penalty,
                                              capacitated=capacitated)
        self._cache: dict[bytes, EvaluationResult] = {}
        self._n_lp_solves = 0
        # >>> CHANGED: the references live on the Evaluator. Your version wrote them into the instance the
        # first time, so a second Evaluator with another omega silently reused stale numbers. They are
        # computed with the SAME omega as the follower (PDF 9.3). Explicit references on the instance
        # (set_references) are still respected. <<<
        if instance.c_op_ref is not None and instance.c_pass_ref is not None:
            self.c_op_ref, self.c_pass_ref = instance.c_op_ref, instance.c_pass_ref
        else:
            self.c_op_ref, self.c_pass_ref = compute_reference_values(instance, self.unmet_penalty)

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
                                      None, None, None, violation=self._violation(x))
        else:
            assignment = self.assignment.solve(x, self.frequencies)
            self._n_lp_solves += 1
            if not assignment.feasible:
                result = EvaluationResult(x.copy(), False, "assignment_failed", c_op,
                                          np.inf, np.inf, None, None, assignment)
            else:
                passenger_cost = assignment.objective
                objective = (
                    self.instance.lambda_op * c_op / self.c_op_ref
                    + self.instance.lambda_pass * passenger_cost / self.c_pass_ref
                )
                tt, weights = passenger_travel_times(self.instance, assignment.flows)
                gini = weighted_gini(tt, weights)
                p95 = weighted_percentile(tt, weights)
                result = EvaluationResult(
                    x.copy(), True, "ok", c_op, passenger_cost, float(objective), gini, p95, assignment,
                    travel_time=assignment.c_pass, unserved=assignment.unserved_total,
                    served_fraction=assignment.served_fraction,
                    c_op_norm=c_op / self.c_op_ref, c_pass_norm=passenger_cost / self.c_pass_ref)
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

    # >>> CHANGED: fitness of an infeasible plan. It used to be the constant 1e6 (= omega), so a plan far over
    # budget and a plan just over budget looked equal to the GA. Now it is worse than ANY feasible plan
    # (+ 1 is above the largest feasible objective, see worst_objective) and grows with the violation, so
    # the GA is pushed back towards the feasible region. <<<
    def fitness(self, result: EvaluationResult) -> float:
        """Output: value the GA minimises (objective, or penalised value if infeasible)."""
        if result.feasible:
            return result.objective
        return self.worst_objective + 1.0 + result.violation

    @property
    def worst_objective(self) -> float:
        """Largest objective a feasible plan can have: all-line cost cannot exceed the budget, and the empty
        plan leaves everybody unserved (cost omega * total demand)."""
        inst = self.instance
        c_op_max = min(float(inst.line_cost.sum()), inst.budget)
        c_pass_max = self.unmet_penalty * float(inst.demand.sum())
        return inst.lambda_op * c_op_max / self.c_op_ref + inst.lambda_pass * c_pass_max / self.c_pass_ref

    @property
    def n_lp_solves(self) -> int:
        return self._n_lp_solves

    def _violation(self, x: np.ndarray) -> float:
        """Relative amount by which budget and fleet are exceeded (0 if both hold)."""
        inst = self.instance
        over_budget = max(0.0, float(inst.line_cost @ x) - inst.budget) / max(inst.line_cost.sum(), 1.0)
        over_fleet = max(0.0, float(inst.line_fleet @ x) - inst.fleet_total) / max(inst.line_fleet.sum(), 1.0)
        return over_budget + over_fleet

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


def compute_reference_values(instance: Instance, unmet_penalty: float | None = None) -> tuple[float, float]:
    """Output: (C_op_ref, C_pass_ref) from the all-lines-active plan, passengers ignoring capacity.
    The same omega as in the follower LP is used (PDF 9.3), so both are measured in the same unit."""
    instance.build()
    if unmet_penalty is None:
        unmet_penalty = instance.unmet_penalty if instance.unmet_penalty is not None \
            else instance.default_unmet_penalty()
    x = np.ones(instance.n_lines, dtype=np.uint8)
    frequencies = np.asarray([line.frequency for line in instance.lines], dtype=float)
    assignment = PassengerAssignment(instance, unmet_penalty=unmet_penalty,
                                     capacitated=False).solve(x, frequencies)
    if not assignment.feasible:
        raise RuntimeError(f"reference assignment failed: {assignment.status}")
    # A unit fallback keeps normalization defined for zero-cost toy instances.
    return max(float(instance.line_cost.sum()), 1.0), max(float(assignment.objective), 1.0)
