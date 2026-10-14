"""Panel 1: the transit network for one plan.

>>> CHANGED (was two stubs). Reading the picture: grey = streets without service, coloured = a link served by
an active line. Line WIDTH = passengers carried, line COLOUR = how full the vehicles are (white -> dark red),
dot size = how many people start or end their trip there. <<<
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..assignment import AssignmentResult
from ..instance import Instance


def node_positions(instance: Instance, seed: int = 0) -> dict[int, tuple[float, float]]:
    """Output: node id -> (x, y). Uses nodes.x/y if present, else a fixed layout computed from the links
    (so the picture does not jump between generations)."""
    nodes = instance.nodes
    if {"x", "y"}.issubset(nodes.columns) and nodes[["x", "y"]].notna().all().all():
        return {int(r.id): (float(r.x), float(r.y)) for r in nodes.itertuples()}
    graph = nx.Graph()
    graph.add_nodes_from(nodes["id"])
    if instance.edges is not None:
        graph.add_weighted_edges_from(instance.edges[["u", "v", "time"]].itertuples(index=False))
    return {int(k): (float(v[0]), float(v[1])) for k, v in nx.kamada_kawai_layout(graph, weight="weight").items()}


def _link(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def plot_plan(instance: Instance, x: np.ndarray, assignment: AssignmentResult | None,
              ax: plt.Axes, pos: dict | None = None, title: str = "") -> None:
    """Input: plan x and its y*(x) result. Output: modifies ax."""
    pos = pos or node_positions(instance)
    x = np.asarray(x)
    ax.clear()
    # 1. streets (grey)
    if instance.edges is not None:
        for u, v in {_link(a, b) for a, b in zip(instance.edges["u"], instance.edges["v"])}:
            ax.plot(*zip(pos[u], pos[v]), color="#d0d0d0", lw=1.5, zorder=1)
    # 2. load and capacity per physical link, summed over both directions
    load, capacity = {}, {}
    for line_pos, line in enumerate(instance.lines):
        if not x[line_pos]:
            continue
        for s, (u, v) in enumerate(line.segments):
            row = instance.seg_index[line.id, s]
            key = _link(u, v)
            capacity[key] = capacity.get(key, 0.0) + instance.seg_capacity[row]
            load[key] = load.get(key, 0.0) + (assignment.segment_load[row] if assignment is not None
                                              and assignment.segment_load is not None else 0.0)
    biggest = max(load.values(), default=0.0) or 1.0
    colours = plt.get_cmap("YlOrRd")
    for key, cap in capacity.items():
        utilisation = min(load[key] / cap, 1.0) if cap > 0 else 0.0
        ax.plot(*zip(pos[key[0]], pos[key[1]]), color=colours(0.1 + 0.9 * utilisation),
                lw=1.5 + 9.0 * load[key] / biggest, solid_capstyle="round", zorder=2)
    # 3. stops
    size = instance.node_demand()
    for node, (px, py) in pos.items():
        ax.scatter(px, py, s=40 + 260 * size.get(node, 0.0) / max(size.max(), 1.0), color="white",
                   edgecolor="black", zorder=3)
        ax.annotate(str(node), (px, py), ha="center", va="center", fontsize=7, zorder=4)
    ax.set_title(title or "Best plan so far", fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")
