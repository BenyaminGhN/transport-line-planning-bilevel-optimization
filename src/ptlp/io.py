"""File reading (and the small writers that go with it). No optimisation lives here.

>>> CHANGED (whole file, it was three stubs). One folder = one study area:

    nodes.csv            id, x, y            (Mandl: mandl1_nodes.txt, x = lon, y = lat)
    edges.csv            u, v, time          one row per link; the reverse direction is added automatically
                                             (Mandl: mandl1_links.txt already lists both directions)
    od_demand.csv        origin, dest, demand   passengers per horizon (Mandl: mandl1_demand.txt)
    candidate_lines.csv  written by lines.save_lines (stops, times, cost, frequency, capacity, fleet, two-way)
    params.json          horizon, budget, fleet_total, lambda_op, lambda_pass, unmet_penalty (null = default),
                         k_routes, max_transfers, detour_factor
<<<
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .instance import Instance
from .lines import load_lines
from .routes import attach_routes, generate_routes

# File names per data set: (nodes, edges, od demand)
_FILES = {
    "synthetic": ("nodes.csv", "edges.csv", "od_demand.csv"),
    "mandl": ("mandl1_nodes.txt", "mandl1_links.txt", "mandl1_demand.txt"),
    # >>> NEW: 20 Dutch cities (your fake-data generator), converted by datagen.import_network <<<
    "netherlands": ("nodes.csv", "edges.csv", "od_demand.csv"),
}
DEFAULT_PARAMS = {
    "horizon": 1.0, "budget": None, "fleet_total": None, "lambda_op": 0.5, "lambda_pass": 0.5,
    "unmet_penalty": None, "k_routes": 3, "max_transfers": 2, "detour_factor": 2.5,
}


def _read_tables(folder: Path, files: tuple[str, str, str]):
    nodes = pd.read_csv(folder / files[0])
    edges = pd.read_csv(folder / files[1]).rename(columns={"from": "u", "to": "v", "travel_time": "time"})
    od = pd.read_csv(folder / files[2]).rename(columns={"from": "origin", "to": "dest"})
    if "lon" in nodes.columns:                                   # Mandl file: map coordinates -> x, y
        nodes = nodes.rename(columns={"lon": "x", "lat": "y"})
    nodes = nodes[[c for c in ("id", "x", "y") if c in nodes.columns]]
    edges = _both_directions(edges[["u", "v", "time"]])
    od = od[od["origin"] != od["dest"]].reset_index(drop=True)
    od.insert(0, "k", range(len(od)))
    return nodes, edges, od[["k", "origin", "dest", "demand"]]


def _both_directions(edges: pd.DataFrame) -> pd.DataFrame:
    """Add the reverse arc of every link that does not list it yet (links are two-way streets)."""
    known = set(zip(edges["u"], edges["v"]))
    extra = edges[[(v, u) not in known for u, v in zip(edges["u"], edges["v"])]]
    extra = extra.rename(columns={"u": "v", "v": "u"})[["u", "v", "time"]]
    return pd.concat([edges, extra], ignore_index=True)


def read_params(folder: Path) -> dict:
    """params.json merged over DEFAULT_PARAMS."""
    params = dict(DEFAULT_PARAMS)
    path = folder / "params.json"
    if path.exists():
        params.update(json.loads(path.read_text()))
    return params


def _make_instance(name: str, folder: Path, files: tuple[str, str, str]) -> Instance:
    nodes, edges, od = _read_tables(folder, files)
    params = read_params(folder)
    lines_path = folder / "candidate_lines.csv"
    if not lines_path.exists():
        raise FileNotFoundError(f"{lines_path} is missing - run `python -m scripts.generate_data` first")
    lines = load_lines(lines_path)
    total_cost = sum(line.cost for line in lines)
    total_fleet = sum(line.fleet for line in lines)
    return Instance(
        name=name, nodes=nodes, od=od, lines=lines, edges=edges,
        horizon=float(params["horizon"]),
        budget=float(params["budget"]) if params["budget"] is not None else total_cost,
        fleet_total=int(params["fleet_total"]) if params["fleet_total"] is not None else total_fleet,
        lambda_op=float(params["lambda_op"]), lambda_pass=float(params["lambda_pass"]),
        unmet_penalty=params["unmet_penalty"],
    )


def load_synthetic(folder: Path) -> Instance:
    """Input: data/synthetic/. Output: Instance with nodes, od, lines (routes still empty)."""
    return _make_instance("synthetic", Path(folder), _FILES["synthetic"])


def load_mandl(folder: Path) -> Instance:
    """Input: data/mandl/. Output: Instance (same contract as load_synthetic)."""
    return _make_instance("mandl", Path(folder), _FILES["mandl"])


def load_instance(name: str, root: Path = Path("data"), k_routes: int | None = None) -> Instance:
    """Input: 'synthetic' | 'mandl' | 'netherlands'. Output: fully prepared Instance (lines -> routes -> build() done).
    k_routes overrides the value of params.json (used by the route-set sensitivity)."""
    if name not in _FILES:
        raise ValueError(f"unknown data set {name!r}; use one of {sorted(_FILES)}")  # >>> CHANGED <<<
    folder = Path(root) / name
    instance = _make_instance(name, folder, _FILES[name])
    params = read_params(folder)
    routes = generate_routes(instance, k_routes=k_routes or int(params["k_routes"]),
                             max_transfers=int(params["max_transfers"]),
                             detour_factor=float(params["detour_factor"]))
    attach_routes(instance, routes)
    return instance
