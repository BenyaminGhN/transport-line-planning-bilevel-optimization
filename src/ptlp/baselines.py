"""Baselines of PDF section 17. >>> NEW FILE <<<

17.1 cost-only      : cheapest plan (optionally: cheapest plan that still serves `min_served` of the demand)
17.2 uncapacitated  : best plan when passengers ignore vehicle capacity, then judged WITH capacity
17.3 Stackelberg    : the normal model (exact.py / ga.py)

Note on 17.1: in the soft model an empty plan costs nothing, so the literal cost-only plan is x = 0.
`min_served` (e.g. 0.9) gives a more useful comparison: the cheapest plan that serves at least 90 % of the demand.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluate import Evaluator, EvaluationResult
from .exact import ExactResult, enumerate_all
from .ga import GAConfig, GeneticAlgorithm
from .instance import Instance


def cost_only_baseline(evaluator: Evaluator, exact: ExactResult, min_served: float = 0.0) -> EvaluationResult:
    """Cheapest feasible plan with served share >= min_served (ties: more service, then binary order)."""
    table = exact.table
    ok = table[table["feasible"] & (table["served_fraction"] >= min_served - 1e-9)]
    if ok.empty:
        raise ValueError("no plan serves that much demand")
    best = ok.sort_values(["c_op", "served_fraction"], ascending=[True, False], kind="stable").iloc[0]
    return evaluator.evaluate(np.array([int(c) for c in best["bits"]], dtype=np.uint8))


@dataclass
class UncapacitatedResult:
    plan: EvaluationResult        # chosen when passengers ignore capacity (its numbers are optimistic)
    under_capacity: EvaluationResult   # the same plan, evaluated with capacity (the truth)


def uncapacitated_baseline(instance: Instance, unmet_penalty: float | None = None,
                           max_exact_lines: int = 16, seed: int = 0) -> UncapacitatedResult:
    """Choose the plan with a capacity-blind follower, then re-evaluate it with the capacity-aware follower."""
    blind = Evaluator(instance, capacitated=False, unmet_penalty=unmet_penalty)
    real = Evaluator(instance, capacitated=True, unmet_penalty=unmet_penalty)
    # the objective must be normalised the same way in both worlds, otherwise the numbers cannot be compared
    blind.c_op_ref, blind.c_pass_ref = real.c_op_ref, real.c_pass_ref
    if instance.n_lines <= max_exact_lines:
        plan = enumerate_all(blind, max_lines=max_exact_lines).best
    else:
        plan = GeneticAlgorithm(blind, GAConfig(seed=seed)).run().best
    return UncapacitatedResult(plan, real.evaluate(plan.x))


def comparison_table(rows: dict[str, EvaluationResult]) -> pd.DataFrame:
    """One line per method with the numbers reported in the thesis."""
    return pd.DataFrame([{
        "method": name, "plan": "".join(map(str, r.x)), "lines_on": int(r.x.sum()),
        "objective": r.objective, "c_op": r.c_op, "ride_time": r.travel_time, "served_fraction": r.served_fraction,
        "gini": r.gini, "p95": r.p95,
    } for name, r in rows.items()])
