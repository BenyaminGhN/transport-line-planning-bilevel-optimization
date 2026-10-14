"""Exhaustive enumeration -> exact benchmark for small |L|."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter

import numpy as np
import pandas as pd

from .evaluate import Evaluator, EvaluationResult


@dataclass
class ExactResult:
    best: EvaluationResult
    table: pd.DataFrame        # one row per plan, see the columns below
    runtime_s: float
    n_lp_solves: int = 0       # >>> NEW: LPs solved by the enumeration (budget/fleet-infeasible plans need none) <<<


def enumerate_all(evaluator: Evaluator, max_lines: int = 20) -> ExactResult:
    """Input: Evaluator. Output: ExactResult over all 2^m plans.
    Raises if m > max_lines. The table also feeds the dashboard (grey 'all plans' cloud).
    >>> CHANGED: table has more columns: travel_time, unserved, served_fraction, c_op_norm, c_pass_norm
    (the normalised parts of the objective), so every plot of the thesis can be made from this one table. <<<"""
    n_lines = evaluator.instance.n_lines
    if n_lines > max_lines:
        raise ValueError(f"exact enumeration is limited to {max_lines} lines")
    started = perf_counter()
    solves_before = evaluator.n_lp_solves
    rows: list[dict] = []
    best: EvaluationResult | None = None
    for bits in product((0, 1), repeat=n_lines):
        x = np.asarray(bits, dtype=np.uint8)
        result = evaluator.evaluate(x)
        rows.append({
            "bits": "".join(map(str, bits)),
            "n_lines_on": int(x.sum()),
            "c_op": result.c_op,
            "c_pass": result.c_pass,
            "objective": result.objective,
            "feasible": result.feasible,
            "reason": result.reason,
            "gini": result.gini,
            "p95": result.p95,
            "travel_time": result.travel_time,
            "unserved": result.unserved,
            "served_fraction": result.served_fraction,
            "c_op_norm": result.c_op_norm,
            "c_pass_norm": result.c_pass_norm,
        })
        # strict '<' and the fixed enumeration order make the benchmark deterministic: of several plans with
        # the same objective, the first one in binary order wins (PDF 13.7 / 18.6)
        if result.feasible and (best is None or result.objective < best.objective - 1e-12):
            best = result
    if best is None:
        raise RuntimeError("enumeration found no feasible line plan")
    return ExactResult(best, pd.DataFrame(rows), perf_counter() - started,
                       evaluator.n_lp_solves - solves_before)
