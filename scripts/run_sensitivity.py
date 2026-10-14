"""Sensitivity analysis of PDF section 18.   >>> CHANGED: implemented <<<

Run from the repository root:
    python -m scripts.run_sensitivity                 # all sweeps on the synthetic network (about 1 minute)
    python -m scripts.run_sensitivity --only capacity omega
Each sweep re-solves the whole game (exact enumeration) and writes results/sensitivity/<parameter>.csv + .png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ptlp.evaluate import Evaluator
from ptlp.exact import enumerate_all
from ptlp.ga import GAConfig, GeneticAlgorithm
from ptlp.io import load_instance

# parameter -> (values, what the number means)
SWEEPS = {
    "capacity":    ([0.5, 0.6, 0.7, 0.8, 1.0, 1.25, 1.5], "vehicle capacity x factor"),
    "lambda_op":   ([0.1, 0.3, 0.5, 0.7, 0.9], "weight of operator cost"),
    "omega":       ([1, 2, 5, 10, 20], "unserved penalty = factor x longest route"),
    "budget":      ([0.3, 0.45, 0.6, 0.8, 1.0], "budget as share of all line costs"),
    "fleet":       ([0.3, 0.45, 0.6, 0.8, 1.0], "fleet as share of all vehicles"),
    "k_routes":    ([1, 2, 3, 5, 8], "routes per OD pair"),
    "capacitated": ([0, 1], "0 = passengers ignore capacity (result judged WITH capacity)"),
}


def solve(instance, unmet_penalty=None, capacitated=True, method="exact", seed=0):
    """Best plan of one game. Returns its EvaluationResult."""
    evaluator = Evaluator(instance, capacitated=capacitated, unmet_penalty=unmet_penalty)
    if method == "exact":
        return enumerate_all(evaluator, max_lines=max(20, instance.n_lines)).best
    return GeneticAlgorithm(evaluator, GAConfig(seed=seed)).run().best


def sweep(instance_name: str, parameter: str, values: list[float], method: str = "exact") -> pd.DataFrame:
    """Input: parameter in SWEEPS. Output: tidy DataFrame (one row per value)."""
    base = load_instance(instance_name)
    total_cost, total_fleet = sum(l.cost for l in base.lines), sum(l.fleet for l in base.lines)
    rows = []
    for value in values:
        instance, omega, capacitated = base, None, True
        if parameter == "capacity":
            instance = base.modified(capacity_scale=value)
        elif parameter == "lambda_op":
            instance = base.modified(lambda_op=value)
        elif parameter == "omega":
            omega = value * float(base.route_cost.max())
        elif parameter == "budget":
            instance = base.modified(budget=value * total_cost)
        elif parameter == "fleet":
            instance = base.modified(fleet_total=value * total_fleet)
        elif parameter == "k_routes":
            instance = load_instance(instance_name, k_routes=int(value))
        elif parameter == "capacitated":
            capacitated = bool(value)
        else:
            raise ValueError(f"unknown parameter {parameter}")
        best = solve(instance, omega, capacitated, method)
        if not capacitated:      # judge the capacity-blind plan with the real follower
            best = Evaluator(instance).evaluate(best.x)
        rows.append({"parameter": parameter, "value": value, "plan": "".join(map(str, best.x)),
                     "lines_on": int(best.x.sum()), "objective": best.objective, "c_op": best.c_op,
                     "ride_time": best.travel_time, "served_fraction": best.served_fraction,
                     "gini": best.gini, "p95": best.p95})
    return pd.DataFrame(rows)


def plot_sweep(table: pd.DataFrame, path: Path, label: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, column, title in zip(axes, ["objective", "served_fraction", "lines_on"],
                                 ["objective", "served share of demand", "lines in the plan"]):
        ax.plot(table["value"], table[column], marker="o")
        ax.set_xlabel(label)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", default="synthetic")
    parser.add_argument("--only", nargs="*", choices=list(SWEEPS), default=list(SWEEPS))
    parser.add_argument("--method", choices=["exact", "ga"], default="exact",
                        help="use 'ga' for networks that are too big to enumerate")
    parser.add_argument("--out", type=Path, default=Path("results/sensitivity"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for parameter in args.only:
        values, label = SWEEPS[parameter]
        table = sweep(args.instance, parameter, values, args.method)
        table.to_csv(args.out / f"{parameter}.csv", index=False)
        plot_sweep(table, args.out / f"{parameter}.png", label)
        print(f"\n== {parameter}: {label}")
        print(table.drop(columns="parameter").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
