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
                             min_stops: int = 3, max_stops: int = 8, seed: int = 0,
                             bidirectional: bool = False) -> list[list[int]]:  # >>> CHANGED: + bidirectional <<<
    """Input: graph, OD table, number of lines wanted.
    Output: list of stop sequences (shortest paths between high-demand OD pairs, see below).
    bidirectional=True: a path and its reverse are the same two-way line (the reverse is not added again)."""
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

    # >>> CHANGED: new selection rule (your version is replaced, the checks above are yours).
    # Why: your loop collected up to n_lines paths for the FIRST (busiest) OD pair before looking at the
    # next one, so with 10 lines all candidates were near-copies for 1-2 OD pairs and most demand nodes
    # were on no line (tested on a 3x4 grid: 2 of 56 OD pairs covered, one demand node uncovered).
    # Now: (1) round robin over the OD pairs, busiest first - round 1 takes every pair's shortest path,
    #      round 2 the second shortest path of every pair, ... until n_lines are found;
    #      (2) add connecting lines until every demand node is on a line and all lines are connected.
    pairs = [(row.origin, row.dest) for row in ranked_od.itertuples(index=False)
             if row.origin in graph and row.dest in graph and row.origin != row.dest]
    sequences: list[list[int]] = []
    seen: set[tuple] = set()

    def add(path: list[int], strict: bool = True) -> bool:
        """Append path unless known (a two-way line equals its reverse) or outside the stop limits."""
        if strict and not min_stops <= len(path) <= max_stops:
            return False
        key = min(tuple(path), tuple(reversed(path))) if bidirectional else tuple(path)
        if key in seen:
            return False
        seen.add(key)
        sequences.append(list(path))
        return True

    streams = {pair: _simple_paths(graph, *pair) for pair in pairs}   # shortest path first, then longer
    active = list(pairs)
    while active and len(sequences) < n_lines:
        still_active = []
        for pair in active:
            if len(sequences) >= n_lines:
                break
            path = _next_new_path(streams[pair], add)
            if path is not None:
                still_active.append(pair)
        active = still_active
    _add_connecting_lines(graph, sequences, od, add)
    return sequences
    # <<< END CHANGED


# >>> NEW: helpers of generate_candidate_lines
def _simple_paths(graph: nx.DiGraph, origin, dest):
    """Loop-free paths from origin to dest, shortest first (empty if there is none)."""
    try:
        yield from nx.shortest_simple_paths(graph, origin, dest, weight="time")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return


def _next_new_path(stream, add, max_tries: int = 200):
    """Take paths from the stream until add() accepts one. Returns it, or None if the stream is used up."""
    for _, path in zip(range(max_tries), stream):
        if add(path):
            return path
    return None


def _add_connecting_lines(graph: nx.DiGraph, sequences: list, od: pd.DataFrame, add) -> None:
    """Add shortest-path lines until every demand node is on a line and the lines form one network.
    (So the returned list can be a few lines longer than n_lines.)"""
    demand_nodes = {n for n in set(od["origin"]) | set(od["dest"]) if n in graph}
    dist = dict(nx.all_pairs_dijkstra_path_length(graph, weight="time"))
    for _ in range(len(graph) + 1):                       # each round joins two groups of nodes
        links = nx.Graph()
        links.add_nodes_from(demand_nodes)
        for stops in sequences:
            links.add_edges_from(zip(stops[:-1], stops[1:]))
        groups = sorted(nx.connected_components(links), key=len, reverse=True)
        if len(groups) <= 1:
            return
        best = None                                        # closest pair (main group, other group)
        for other in groups[1:]:
            for u in groups[0]:
                for v in other:
                    for a, b in ((u, v), (v, u)):
                        if b in dist.get(a, {}) and (best is None or dist[a][b] < best[0]):
                            best = (dist[a][b], a, b)
        if best is None:                                   # the network itself is disconnected
            return
        add(nx.shortest_path(graph, best[1], best[2], weight="time"), strict=False)
# <<< END NEW


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
            # >>> NEW: optional column 'bidirectional' (default: one-way, as before) <<<
            bidirectional=bool(row["bidirectional"]) if "bidirectional" in params.columns else False,
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
        "bidirectional": int(line.bidirectional),  # >>> NEW <<<
    } for line in lines]
    pd.DataFrame(records, columns=[
        "id", "stops", "segment_times", "cost", "frequency", "vehicle_capacity", "fleet",
        "bidirectional",  # >>> NEW <<<
    ]).to_csv(path, index=False)


def load_lines(path: str | Path) -> list[Line]:
    """Load candidate lines written by :func:`save_lines`."""
    table = pd.read_csv(path)
    required = {"id", "stops", "segment_times", "cost", "frequency", "vehicle_capacity", "fleet"}
    if not required.issubset(table.columns):
        raise ValueError("candidate-line CSV has missing columns")
    # >>> NEW: 'bidirectional' is optional so files written by your old save_lines still load <<<
    if "bidirectional" not in table.columns:
        table["bidirectional"] = 0
    lines = [Line(
        id=int(row.id),
        stops=tuple(json.loads(row.stops)),
        segment_times=tuple(float(value) for value in json.loads(row.segment_times)),
        cost=float(row.cost),
        frequency=float(row.frequency),
        vehicle_capacity=float(row.vehicle_capacity),  # >>> CHANGED: int() -> float() (int() cut off 28.5) <<<
        fleet=int(row.fleet),
        bidirectional=bool(row.bidirectional),  # >>> NEW <<<
    ) for row in table.itertuples(index=False)]
    if len({line.id for line in lines}) != len(lines):
        raise ValueError("line IDs must be unique")
    return lines
