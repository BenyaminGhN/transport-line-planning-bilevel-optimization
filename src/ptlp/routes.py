"""Passenger-route generation. Run once on the FULL line set; availability is filtered later via x."""
from __future__ import annotations

import networkx as nx
import numpy as np

from .instance import Instance, Line, Route


def build_line_layered_graph(lines: list[Line]) -> nx.DiGraph:
    """Input: all candidate lines.
    Output: graph whose nodes are (stop, line) pairs; ride arcs carry in-vehicle time,
    transfer arcs (same stop, different line) carry cost 0 (waiting/transfer excluded)."""
    graph = nx.DiGraph()
    stop_layers: dict[int, list[tuple[int, int]]] = {}
    for line in lines:
        for stop in line.stops:
            node = (stop, line.id)
            graph.add_node(node, stop=stop, line_id=line.id)
            stop_layers.setdefault(stop, []).append(node)
        for segment_idx, ((u, v), travel_time) in enumerate(zip(line.segments, line.segment_times)):
            graph.add_edge((u, line.id), (v, line.id), weight=float(travel_time),
                           kind="ride", leg=(line.id, segment_idx))

    for layers in stop_layers.values():
        for source in layers:
            for target in layers:
                if source != target:
                    graph.add_edge(source, target, weight=0.0, kind="transfer", leg=None)
    return graph


def generate_routes(instance: Instance, k_routes: int = 3) -> list[Route]:
    """Input: Instance (lines + od). Output: up to k_routes loop-free routes per OD pair
    (k-shortest paths in the layered graph), each with legs and cost C_kr."""
    if k_routes < 1:
        raise ValueError("k_routes must be positive")
    required = {"k", "origin", "dest", "demand"}
    if not required.issubset(instance.od.columns):
        raise ValueError("instance.od must contain k, origin, dest and demand")

    base = build_line_layered_graph(instance.lines)
    routes: list[Route] = []
    next_id = 0
    for od_row in instance.od.itertuples(index=False):
        origin_layers = [node for node, data in base.nodes(data=True) if data["stop"] == od_row.origin]
        destination_layers = [node for node, data in base.nodes(data=True) if data["stop"] == od_row.dest]
        if not origin_layers or not destination_layers or od_row.origin == od_row.dest:
            continue

        graph = base.copy()
        source = ("__source__", od_row.k)
        target = ("__target__", od_row.k)
        graph.add_node(source)
        graph.add_node(target)
        for node in origin_layers:
            graph.add_edge(source, node, weight=0.0, kind="access", leg=None)
        for node in destination_layers:
            graph.add_edge(node, target, weight=0.0, kind="egress", leg=None)

        seen_legs: set[tuple[tuple[int, int], ...]] = set()
        try:
            paths = nx.shortest_simple_paths(graph, source, target, weight="weight")
            for candidate_index, path in enumerate(paths):
                legs: list[tuple[int, int]] = []
                physical_stops = [od_row.origin]
                cost = 0.0
                for u, v in zip(path[:-1], path[1:]):
                    edge = graph[u][v]
                    if edge.get("kind") == "ride":
                        legs.append(edge["leg"])
                        physical_stops.append(v[0])
                        cost += float(edge["weight"])
                key = tuple(legs)
                if (not key or key in seen_legs
                        or len(set(physical_stops)) != len(physical_stops)):
                    if candidate_index >= max(100, 50 * k_routes):
                        break
                    continue
                seen_legs.add(key)
                routes.append(Route(next_id, od_row.k, key, cost))
                next_id += 1
                if len(seen_legs) >= k_routes:
                    break
                if candidate_index >= max(100, 50 * k_routes):
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    return routes


def attach_routes(instance: Instance, routes: list[Route]) -> None:
    """Store routes on the instance and call instance.build()."""
    instance.routes = list(routes)
    instance.build()
