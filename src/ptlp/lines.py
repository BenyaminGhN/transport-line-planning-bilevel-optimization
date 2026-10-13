"""Candidate-line generation. Run once per instance, then save to candidate_lines.csv."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .instance import Line


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """Input: nodes (id,x,y), edges (u,v,time). Output: directed graph, edge attr 'time'."""
    required_nodes = {"id"}
    required_edges = {"u", "v", "time"}
    if not required_nodes.issubset(nodes.columns):
        raise ValueError("nodes must contain an 'id' column")
    if not required_edges.issubset(edges.columns):
        raise ValueError("edges must contain 'u', 'v' and 'time' columns")
    if nodes["id"].duplicated().any():
        raise ValueError("node IDs must be unique")

    graph = nx.DiGraph()
    for record in nodes.to_dict(orient="records"):
        node_id = record.pop("id")
        graph.add_node(node_id, **record)

    for record in edges.to_dict(orient="records"):
        u, v, travel_time = record.pop("u"), record.pop("v"), record.pop("time")
        if u not in graph or v not in graph:
            raise ValueError("every edge endpoint must reference a known node")
        if not np.isfinite(travel_time) or travel_time < 0:
            raise ValueError("edge times must be finite and nonnegative")
        if graph.has_edge(u, v):
            raise ValueError(f"duplicate directed edge ({u}, {v})")
        graph.add_edge(u, v, time=float(travel_time), **record)
    return graph


def generate_candidate_lines(graph: nx.DiGraph, od: pd.DataFrame, n_lines: int,
                             min_stops: int = 3, max_stops: int = 8, seed: int = 0) -> list[list[int]]:
    """Input: graph, OD table, number of lines wanted.
    Output: list of stop sequences (e.g. shortest paths between high-demand OD pairs)."""
    del seed  # The ranking is deliberately deterministic; retained for API compatibility.
    if n_lines < 0:
        raise ValueError("n_lines must be nonnegative")
    if min_stops < 2 or max_stops < min_stops:
        raise ValueError("require 2 <= min_stops <= max_stops")
    required = {"origin", "dest", "demand"}
    if not required.issubset(od.columns):
        raise ValueError("od must contain 'origin', 'dest' and 'demand' columns")
    if n_lines == 0:
        return []

    ranked_od = od.reset_index(drop=True).copy()
    demand = ranked_od["demand"].to_numpy(dtype=float)
    if np.any(~np.isfinite(demand)) or np.any(demand < 0):
        raise ValueError("OD demand must be finite and nonnegative")
    ranked_od["_row"] = np.arange(len(ranked_od))
    ranked_od = ranked_od.sort_values(["demand", "_row"], ascending=[False, True])

    candidates: list[tuple[float, float, tuple]] = []
    seen: set[tuple] = set()
    per_od_limit = max(n_lines, 1)
    for row in ranked_od.itertuples(index=False):
        origin, dest = row.origin, row.dest
        if origin not in graph or dest not in graph or origin == dest:
            continue
        try:
            paths = nx.shortest_simple_paths(graph, origin, dest, weight="time")
            accepted = 0
            for candidate_index, path in enumerate(paths):
                if candidate_index >= max(100, 50 * per_od_limit):
                    break
                if len(path) > max_stops:
                    continue
                if len(path) < min_stops:
                    continue
                key = tuple(path)
                if key in seen:
                    continue
                seen.add(key)
                travel_time = nx.path_weight(graph, path, weight="time")
                candidates.append((-float(row.demand), float(travel_time), key))
                accepted += 1
                if accepted >= per_od_limit:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    candidates.sort(key=lambda item: (item[0], item[1], tuple(map(str, item[2]))))
    return [list(path) for _, _, path in candidates[:n_lines]]


def attach_line_parameters(stop_sequences: list[list[int]], graph: nx.DiGraph,
                           params: pd.DataFrame) -> list[Line]:
    """Input: stop sequences + table (cost, frequency, vehicle_capacity, fleet).
    Output: list[Line] with segment_times taken from the graph."""
    required = {"cost", "frequency", "vehicle_capacity", "fleet"}
    if not required.issubset(params.columns):
        raise ValueError("params must contain cost, frequency, vehicle_capacity and fleet")
    if len(params) != len(stop_sequences):
        raise ValueError("params must have one row per stop sequence")

    lines: list[Line] = []
    for position, (stops, (_, row)) in enumerate(zip(stop_sequences, params.iterrows())):
        if len(stops) < 2:
            raise ValueError("each line must contain at least two stops")
        try:
            segment_times = tuple(float(graph[u][v]["time"]) for u, v in zip(stops[:-1], stops[1:]))
        except KeyError as exc:
            raise ValueError("a stop sequence uses an edge missing from the graph") from exc
        line_id = row["id"] if "id" in params.columns else position
        values = np.asarray([row["cost"], row["frequency"], row["vehicle_capacity"], row["fleet"]],
                            dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("line parameters must be finite and nonnegative")
        if not float(row["vehicle_capacity"]).is_integer() or not float(row["fleet"]).is_integer():
            raise ValueError("vehicle_capacity and fleet must be integers")
        lines.append(Line(
            id=int(line_id),
            stops=tuple(stops),
            segment_times=segment_times,
            cost=float(row["cost"]),
            frequency=float(row["frequency"]),
            vehicle_capacity=int(row["vehicle_capacity"]),
            fleet=int(row["fleet"]),
        ))
    if len({line.id for line in lines}) != len(lines):
        raise ValueError("line IDs must be unique")
    return lines


def save_lines(lines: list[Line], path: str | Path) -> None:
    """Save candidate lines to a stable CSV representation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [{
        "id": line.id,
        "stops": json.dumps(list(line.stops)),
        "segment_times": json.dumps(list(line.segment_times)),
        "cost": line.cost,
        "frequency": line.frequency,
        "vehicle_capacity": line.vehicle_capacity,
        "fleet": line.fleet,
    } for line in lines]
    pd.DataFrame(records, columns=[
        "id", "stops", "segment_times", "cost", "frequency", "vehicle_capacity", "fleet"
    ]).to_csv(path, index=False)


def load_lines(path: str | Path) -> list[Line]:
    """Load candidate lines written by :func:`save_lines`."""
    table = pd.read_csv(path)
    required = {"id", "stops", "segment_times", "cost", "frequency", "vehicle_capacity", "fleet"}
    if not required.issubset(table.columns):
        raise ValueError("candidate-line CSV has missing columns")
    lines = [Line(
        id=int(row.id),
        stops=tuple(json.loads(row.stops)),
        segment_times=tuple(float(value) for value in json.loads(row.segment_times)),
        cost=float(row.cost),
        frequency=float(row.frequency),
        vehicle_capacity=int(row.vehicle_capacity),
        fleet=int(row.fleet),
    ) for row in table.itertuples(index=False)]
    if len({line.id for line in lines}) != len(lines):
        raise ValueError("line IDs must be unique")
    return lines
