"""Panels 2-4: algorithm behaviour. Each function draws on a given Axes.

>>> CHANGED (was four stubs). plot_objective_space now takes the exact table (a DataFrame) as grey cloud. <<<
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..ga import GenerationRecord


def plot_convergence(history: list[GenerationRecord], ax: plt.Axes,
                     exact_objective: float | None = None) -> None:
    """Best-so-far, population mean and worst feasible fitness vs generation;
    dashed horizontal line at the exact optimum when known."""
    ax.clear()
    generations = [r.generation for r in history]
    feasible = [r.fitness[r.feasible] for r in history]
    ax.plot(generations, [r.best_fitness for r in history], color="tab:blue", lw=2, label="best so far")
    ax.plot(generations, [f.mean() if len(f) else np.nan for f in feasible], color="tab:orange", label="population mean")
    ax.plot(generations, [f.max() if len(f) else np.nan for f in feasible], color="tab:gray", ls=":", label="worst feasible")
    if exact_objective is not None:
        ax.axhline(exact_objective, color="green", ls="--", label="exact optimum")
    ax.set_xlabel("generation")
    ax.set_ylabel("objective (lower is better)")
    ax.set_title("Convergence", fontsize=9)
    ax.legend(fontsize=7)


def plot_objective_space(record: GenerationRecord, ax: plt.Axes,
                         all_plans: pd.DataFrame | None = None,
                         exact_best: tuple[float, float] | None = None) -> None:
    """Scatter of the current population in (operator cost, passenger cost) space, colour = served share.
    Optional grey cloud = all enumerated plans, star = exact optimum, ring = GA best."""
    ax.clear()
    if all_plans is not None:
        ok = all_plans[all_plans["feasible"]]
        ax.scatter(ok["c_op_norm"], ok["c_pass_norm"], s=8, color="#dddddd", zorder=1, label="all plans")
    ok = record.feasible
    points = ax.scatter(record.c_op[ok], record.c_pass[ok], c=record.served_fraction[ok], cmap="viridis",
                        vmin=0, vmax=1, s=28, edgecolor="black", linewidth=0.3, zorder=2, label="population")
    best = record.best_result
    ax.scatter([best.c_op_norm], [best.c_pass_norm], s=140, facecolor="none", edgecolor="red", lw=2,
               zorder=3, label="GA best")
    if exact_best is not None:
        ax.scatter(*exact_best, marker="*", s=160, color="green", zorder=4, label="exact optimum")
    ax.set_xlabel("operator cost / reference")
    ax.set_ylabel("passenger cost / reference")
    ax.set_title(f"Objective space ({int((~ok).sum())} infeasible not shown)", fontsize=9)
    ax.legend(fontsize=7)
    return points


def plot_population_matrix(record: GenerationRecord, ax: plt.Axes, line_ids: list[int]) -> None:
    """Heat-map: rows = individuals sorted by fitness (best on top), columns = lines, cell = x_l.
    Infeasible rows are marked with a cross on the left. Shows how the genes converge."""
    ax.clear()
    order = np.argsort(record.fitness, kind="stable")
    ax.imshow(record.population[order], cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for row, individual in enumerate(order):
        if not record.feasible[individual]:
            ax.text(-0.9, row, "x", color="red", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(line_ids)))
    ax.set_xticklabels(line_ids, fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("line id")
    ax.set_title("Population (dark = line on)", fontsize=9)


def plot_diversity_and_cost(history: list[GenerationRecord], ax: plt.Axes) -> None:
    """Diversity and cumulative LP solves vs generation (twin axes)."""
    ax.clear()
    generations = [r.generation for r in history]
    ax.plot(generations, [r.diversity for r in history], color="tab:purple")
    ax.set_xlabel("generation")
    ax.set_ylabel("diversity", color="tab:purple")
    twin = ax.twinx()
    twin.plot(generations, [r.n_lp_solves for r in history], color="tab:red")
    twin.set_ylabel("LP solves", color="tab:red")
