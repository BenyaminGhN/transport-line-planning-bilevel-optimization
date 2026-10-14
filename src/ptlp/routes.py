"""Passenger-route generation. Run once on the FULL line set; availability is filtered later via x."""
from __future__ import annotations

from collections import defaultdict

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
        # >>> CHANGED: loop over the segment indexes so that the reverse segments of two-way lines are
        # included (zip(line.segments, line.segment_times) would stop after the forward half). <<<
        for segment_idx, (u, v) in enumerate(line.segments):
            graph.add_edge((u, line.id), (v, line.id), weight=float(line.segment_time(segment_idx)),
                           kind="ride", leg=(line.id, segment_idx))

    for layers in stop_layers.values():
        for source in layers:
            for target in layers:
                if source != target:
                    graph.add_edge(source, target, weight=0.0, kind="transfer", leg=None)
    return graph


# >>> CHANGED: generate_routes is rewritten (same name, same output type, new default arguments).
# Why: the k-shortest-path search on the layered graph needed 19 s for k_routes=10 on a 12-node, 12-line
# network (about 1 s for k_routes=3) because the zero-cost transfer arcs create thousands of equal-cost
# duplicates. The new version lists the routes directly and needs ~0.1 s. It also chooses WHICH k routes
# to keep, so that small line plans (few lines on) still have a route for most OD pairs.
def generate_routes(instance: Instance, k_routes: int = 3, max_transfers: int = 2,
                    detour_factor: float = 2.5) -> list[Route]:
    """Input: Instance (lines + od). Output: up to k_routes loop-free routes per OD pair, each with legs
    and cost C_kr (in-vehicle time only).

    A route rides a line from the origin, may change line at any stop (at most max_transfers times) and
    must not be slower than detour_factor x the fastest route of the full network.
    An OD pair that no route can serve gets no route (its demand will be unserved)."""
    if k_routes < 1:
        raise ValueError("k_routes must be positive")
    required = {"k", "origin", "dest", "demand"}
    if not required.issubset(instance.od.columns):
        raise ValueError("instance.od must contain k, origin, dest and demand")

    fastest = _fastest_times(build_line_layered_graph(instance.lines))
    lines_at_stop: dict[int, list[tuple[Line, int]]] = defaultdict(list)
    for line in instance.lines:
        for position, stop in enumerate(line.stops):
            lines_at_stop[stop].append((line, position))

    routes: list[Route] = []
    for od_row in instance.od.itertuples(index=False):
        origin, dest = od_row.origin, od_row.dest
        if origin == dest or (origin, dest) not in fastest:
            continue
        found = _enumerate_routes(lines_at_stop, origin, dest, max_transfers,
                                  cost_cap=detour_factor * fastest[origin, dest] + 1e-9)
        for legs, cost in _choose_routes(found, k_routes):
            routes.append(Route(len(routes), od_row.k, legs, cost))
    return routes


def _fastest_times(layered: nx.DiGraph) -> dict[tuple[int, int], float]:
    """Fastest in-vehicle time between every pair of stops that some chain of lines connects."""
    layers: dict[int, list] = defaultdict(list)
    for node, data in layered.nodes(data=True):
        layers[data["stop"]].append(node)
    fastest: dict[tuple[int, int], float] = {}
    for stop, sources in layers.items():
        distance = nx.multi_source_dijkstra_path_length(layered, sources, weight="weight")
        for node, time in distance.items():
            key = (stop, layered.nodes[node]["stop"])
            fastest[key] = min(fastest.get(key, np.inf), time)
    return fastest


def _enumerate_routes(lines_at_stop, origin, dest, max_transfers: int, cost_cap: float) -> dict:
    """All loop-free routes origin -> dest as {legs: in-vehicle time}.
    Board a line at the origin, ride it stop by stop, change line at a stop if transfers are left."""
    found: dict[tuple, float] = {}

    def ride(line: Line, position: int, legs: tuple, time: float, visited: frozenset, transfers: int):
        n = line.n_forward
        for step in ((1, -1) if line.bidirectional else (1,)):
            p, t, leg_list, seen = position, time, legs, visited
            while 0 <= p + step <= n:
                q = p + step
                segment = p if step == 1 else n + q       # forward segment p, or reverse of forward segment q
                t += line.segment_time(segment)
                stop = line.stops[q]
                if t > cost_cap or stop in seen:
                    break
                leg_list, seen, p = leg_list + ((line.id, segment),), seen | {stop}, q
                if stop == dest:
                    found.setdefault(leg_list, t)
                    break
                if transfers < max_transfers:
                    for other, other_position in lines_at_stop[stop]:
                        if other.id != line.id:
                            ride(other, other_position, leg_list, t, seen, transfers + 1)

    for line, position in lines_at_stop.get(origin, []):
        ride(line, position, (), 0.0, frozenset([origin]), 0)
    return found


def _choose_routes(found: dict, k: int) -> list[tuple[tuple, float]]:
    """Keep at most k of the routes of one OD pair, cheapest first. Preference order:
      1. the cheapest route (so the best route of the full network is never lost);
      2. routes whose set of lines is minimal (no other route uses only some of these lines),
         the cheapest one per line set: they keep the OD pair connected when few lines run;
      3. the cheapest route of every other line set;  4. all remaining routes."""
    ranked = sorted(found.items(), key=lambda item: (item[1], item[0]))      # cheapest first
    if not ranked:
        return []
    line_sets = [frozenset(line_id for line_id, _ in legs) for legs, _ in ranked]
    cheapest_of_set: dict[frozenset, int] = {}
    for i, line_set in enumerate(line_sets):
        cheapest_of_set.setdefault(line_set, i)
    representatives = list(cheapest_of_set.values())
    minimal = [i for i in representatives
               if not any(line_sets[j] < line_sets[i] for j in representatives)]
    others = [i for i in representatives if i not in minimal]
    rest = [i for i in range(len(ranked)) if i not in representatives]
    chosen: list[int] = []
    for i in [0] + minimal + others + rest:
        if i not in chosen:
            chosen.append(i)
        if len(chosen) == k:
            break
    return [ranked[i] for i in sorted(chosen)]
# <<< END CHANGED


def attach_routes(instance: Instance, routes: list[Route]) -> None:
    """Store routes on the instance and call instance.build()."""
    instance.routes = list(routes)
    instance.build()
