"""Sensitivity: capacity, lambda weights, budget/fleet, route-set size, baselines."""
from __future__ import annotations
import pandas as pd


def sweep(instance_name: str, parameter: str, values: list[float], seeds: list[int]) -> pd.DataFrame:
    """Input: parameter in {'capacity','lambda_op','budget','fleet','k_routes','capacitated'}.
    Output: tidy DataFrame (parameter, value, seed, objective, x*, c_op, c_pass, gini, p95)."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
