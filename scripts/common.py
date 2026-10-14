"""Shared by run_synthetic.py and run_mandl.py.   >>> NEW FILE <<<"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ptlp.baselines import comparison_table, cost_only_baseline, uncapacitated_baseline
from ptlp.evaluate import Evaluator
from ptlp.exact import enumerate_all
from ptlp.ga import GAConfig, GeneticAlgorithm
from ptlp.io import load_instance
from ptlp.viz.live import LiveDashboard, make_gif
from ptlp.viz.network_plot import plot_plan


def parse_args(description: str, default_exact: bool) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--viz", choices=["off", "live", "save", "both"], default="save",
                   help="dashboard after every generation (save = PNG frames + GIF)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs", type=int, default=5, help="independent GA runs (seed, seed+1, ...)")
    p.add_argument("--pop", type=int, default=30)
    p.add_argument("--gens", type=int, default=40)
    p.add_argument("--lambda-op", type=float, default=None, help="weight of operator cost (default: params.json)")
    p.add_argument("--uncapacitated", action="store_true", help="passengers ignore vehicle capacity")
    p.add_argument("--exact", action=argparse.BooleanOptionalAction, default=default_exact,
                   help="also run the exhaustive enumeration (2^lines plans)")
    p.add_argument("--out", type=Path, default=Path("results"))
    return p.parse_args()


def run_study(name: str, args: argparse.Namespace) -> None:
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    instance = load_instance(name)
    if args.lambda_op is not None:
        instance.lambda_op, instance.lambda_pass = args.lambda_op, 1.0 - args.lambda_op
    capacitated = not args.uncapacitated
    print(f"{name}: {instance.n_lines} lines, {instance.n_od} OD pairs, {instance.n_routes} routes, "
          f"budget {instance.budget}, fleet {instance.fleet_total}")

    exact = None
    if args.exact:
        evaluator = Evaluator(instance, capacitated=capacitated)
        exact = enumerate_all(evaluator, max_lines=max(20, instance.n_lines))
        exact.table.to_csv(out / "all_plans.csv", index=False)
        print(f"exact: best {''.join(map(str, exact.best.x))}  objective {exact.best.objective:.4f}  "
              f"served {exact.best.served_fraction:.1%}  ({exact.n_lp_solves} LPs, {exact.runtime_s:.1f}s)")

    # --- genetic algorithm, several seeds. The dashboard follows the first run only.
    rows, best_run = [], None
    for run in range(args.runs):
        evaluator = Evaluator(instance, capacitated=capacitated)
        dashboard = LiveDashboard(instance, args.viz, out / "frames", exact=exact) if run == 0 else None
        ga = GeneticAlgorithm(evaluator, GAConfig(pop_size=args.pop, n_generations=args.gens, seed=args.seed + run),
                              on_generation=dashboard).run()
        if dashboard is not None:
            dashboard.close()
            if args.viz in ("save", "both"):
                make_gif(out / "frames", out / "ga.gif")
        rows.append({"seed": args.seed + run, "plan": "".join(map(str, ga.best.x)),
                     "objective": ga.best.objective, "served_fraction": ga.best.served_fraction,
                     "lp_solves": ga.n_lp_solves, "seconds": ga.runtime_s,
                     "gap_percent": 100 * (ga.best.objective / exact.best.objective - 1) if exact else np.nan})
        if best_run is None or ga.best.objective < best_run.objective:
            best_run = ga.best
            pd.DataFrame([{"generation": r.generation, "best": r.best_fitness,
                           "mean": r.fitness[r.feasible].mean(), "diversity": r.diversity,
                           "lp_solves": r.n_lp_solves} for r in ga.history]).to_csv(out / "ga_history.csv", index=False)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "ga_runs.csv", index=False)
    print(runs.round(4).to_string(index=False))
    print(f"GA over {len(runs)} runs: objective {runs.objective.mean():.4f} +/- {runs.objective.std(ddof=0):.4f}",
          end="")
    if exact:
        print(f"; gap to exact {runs.gap_percent.mean():.2f}% +/- {runs.gap_percent.std(ddof=0):.2f}%; "
              f"optimum found in {(runs.gap_percent < 1e-9).sum()}/{len(runs)} runs; "
              f"LP solves {runs.lp_solves.mean():.0f} vs {exact.n_lp_solves}")
    else:
        print()

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_plan(instance, best_run.x, best_run.assignment, ax, title=f"Best GA plan {''.join(map(str, best_run.x))}")
    fig.savefig(out / "best_plan.png", dpi=110)
    plt.close(fig)

    if exact and capacitated:
        evaluator = Evaluator(instance)
        blind = uncapacitated_baseline(instance)
        table = comparison_table({
            "cost-only (empty plan)": cost_only_baseline(evaluator, exact),
            "cost-only, >=50% served": cost_only_baseline(evaluator, exact, 0.5) if exact.table.served_fraction.max() >= 0.5 else exact.best,
            "uncapacitated (judged with capacity)": blind.under_capacity,
            "Stackelberg (exact)": exact.best,
        })
        table.to_csv(out / "baselines.csv", index=False)
        print(table.round(3).to_string(index=False))
