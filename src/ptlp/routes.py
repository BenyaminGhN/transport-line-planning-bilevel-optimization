"""Passenger-route generation. Run once on the FULL line set; availability is filtered later via x."""
from __future__ import annotations

import networkx as nx

from .instance import Instance, Line, Route


def build_line_layered_graph(lines: list[Line]) -> nx.DiGraph:
    """Input: all candidate lines.
    Output: graph whose nodes are (stop, line) pairs; ride arcs carry in-vehicle time,
    transfer arcs (same stop, different line) carry cost 0 (waiting/transfer excluded)."""
    raise NotImplementedError


def generate_routes(instance: Instance, k_routes: int = 3) -> list[Route]:
    """Input: Instance (lines + od). Output: up to k_routes loop-free routes per OD pair
    (k-shortest paths in the layered graph), each with legs and cost C_kr."""
    raise NotImplementedError


def attach_routes(instance: Instance, routes: list[Route]) -> None:
    """Store routes on the instance and call instance.build()."""
    raise NotImplementedError
