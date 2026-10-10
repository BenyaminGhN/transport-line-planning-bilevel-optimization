"""Panel 1: the transit network for one plan."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..assignment import AssignmentResult
from ..instance import Instance


def node_positions(instance: Instance, seed: int = 0) -> dict[int, tuple[float, float]]:
    """Output: node id -> (x, y). Uses nodes.x/y if present, else networkx spring layout (fixed seed,
    computed once so the picture does not jump between generations)."""
    raise NotImplementedError


def plot_plan(instance: Instance, x: np.ndarray, assignment: AssignmentResult | None,
              ax: plt.Axes, pos: dict | None = None, title: str = "") -> None:
    """Input: plan x and its y*(x) result.
    Draws: inactive candidate lines in light grey, active lines in distinct colours,
    segment width ~ load, segment colour (white->red) ~ utilisation (load/capacity),
    OD demand as node size. Output: modifies ax."""
    raise NotImplementedError
