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
    table: pd.DataFrame        # one row per plan: bits, c_op, c_pass, objective, feasible, gini, p95
    runtime_s: float


def enumerate_all(evaluator: Evaluator, max_lines: int = 20) -> ExactResult:
    """Input: Evaluator. Output: ExactResult over all 2^m plans.
    Raises if m > max_lines. The table also feeds the dashboard (grey 'all plans' cloud)."""
    n_lines = evaluator.instance.n_lines
    if n_lines > max_lines:
        raise ValueError(f"exact enumeration is limited to {max_lines} lines")
    started = perf_counter()
    rows: list[dict] = []
    best: EvaluationResult | None = None
    for bits in product((0, 1), repeat=n_lines):
        x = np.asarray(bits, dtype=np.uint8)
        result = evaluator.evaluate(x)
        unserved = (float(result.assignment.unserved.sum())
                    if result.assignment is not None and result.assignment.unserved is not None
                    else np.nan)
        rows.append({
            "bits": "".join(map(str, bits)),
            "c_op": result.c_op,
            "c_pass": result.c_pass,
            "objective": result.objective,
            "feasible": result.feasible,
            "reason": result.reason,
            "gini": result.gini,
            "p95": result.p95,
            "unserved": unserved,
        })
        if result.feasible and (best is None or result.objective < best.objective):
            best = result
    if best is None:
        raise RuntimeError("enumeration found no feasible line plan")
    return ExactResult(best, pd.DataFrame(rows), perf_counter() - started)
