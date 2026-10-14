"""Baselines of PDF 17 against the Stackelberg model, for several vehicle capacities.   >>> NEW FILE <<<

Run from the repository root:
    python -m scripts.run_baselines
Writes results/baselines_capacity.csv. With generous capacity all methods agree; the gap opens as capacity shrinks.
"""
from pathlib import Path

import pandas as pd

from ptlp.baselines import comparison_table, cost_only_baseline, uncapacitated_baseline
from ptlp.evaluate import Evaluator
from ptlp.exact import enumerate_all
from ptlp.io import load_instance


def main(name: str = "synthetic", scales=(1.0, 0.7, 0.5), min_served: float = 0.5) -> None:
    tables = []
    for scale in scales:
        instance = load_instance(name).modified(capacity_scale=scale)
        evaluator = Evaluator(instance)
        exact = enumerate_all(evaluator)
        table = comparison_table({
            "cost-only (empty plan)": cost_only_baseline(evaluator, exact),
            f"cost-only, >={min_served:.0%} served": cost_only_baseline(evaluator, exact, min_served),
            "uncapacitated (judged with capacity)": uncapacitated_baseline(instance).under_capacity,
            "Stackelberg (exact)": exact.best,
        })
        table.insert(0, "capacity_factor", scale)
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    Path("results").mkdir(exist_ok=True)
    result.to_csv("results/baselines_capacity.csv", index=False)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
