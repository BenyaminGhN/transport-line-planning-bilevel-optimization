"""Exhaustive enumeration -> exact benchmark for small |L|."""
from __future__ import annotations

from dataclasses import dataclass

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
    raise NotImplementedError
