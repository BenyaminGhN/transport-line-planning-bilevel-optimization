"""Panels 2-4: algorithm behaviour. Each function draws on a given Axes."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..ga import GenerationRecord


def plot_convergence(history: list[GenerationRecord], ax: plt.Axes,
                     exact_objective: float | None = None) -> None:
    """Best-so-far, population mean and worst feasible fitness vs generation;
    dashed horizontal line at the exact optimum when known."""
    raise NotImplementedError


def plot_objective_space(record: GenerationRecord, ax: plt.Axes,
                         all_plans: "np.ndarray | None" = None,
                         exact_best: tuple[float, float] | None = None) -> None:
    """Scatter of the current population in (c_op, c_pass) space.
    Optional grey cloud = all enumerated plans, star = exact optimum, ring = GA best.
    Shows the cost vs travel-time trade-off and where the GA is searching."""
    raise NotImplementedError


def plot_population_matrix(record: GenerationRecord, ax: plt.Axes, line_ids: list[int]) -> None:
    """Heat-map: rows = individuals sorted by fitness, columns = lines, cell = x_l.
    Infeasible rows hatched. Shows convergence of the genes."""
    raise NotImplementedError


def plot_diversity_and_cost(history: list[GenerationRecord], ax: plt.Axes) -> None:
    """Diversity and cumulative LP solves vs generation (twin axes)."""
    raise NotImplementedError
