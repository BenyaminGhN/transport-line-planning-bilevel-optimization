"""Creates the input files of data/synthetic and the candidate lines of data/mandl.

>>> NEW FILE. Run once:  python -m scripts.generate_data
Everything is seeded, so the same call always writes the same files.
<<<
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .assignment import PassengerAssignment
from .instance import Instance, Line
from .io import _make_instance, _FILES, _read_tables, read_params
from .lines import attach_line_parameters, build_graph, generate_candidate_lines, save_lines
from .routes import attach_routes, generate_routes


# ---------------------------------------------------------------- shared helpers
def make_line_table(stop_sequences: list[list[int]], graph, frequency: float = 3.0) -> pd.DataFrame:
    """Simple parameter rules (documented in docs/data_and_experiments.md):
    cost = 2 + 0.5 * one-way length (min);  fleet = vehicles needed for the round trip at `frequency`;
    capacity is a placeholder (calibrate_capacity sets it)."""
    rows = []
    for i, stops in enumerate(stop_sequences):
        length = sum(graph[u][v]["time"] for u, v in zip(stops[:-1], stops[1:]))
        fleet = max(1, int(np.ceil(2 * length / 60.0 * frequency)))
        rows.append({"id": i, "cost": round(2 + 0.5 * length, 1), "frequency": frequency,
                     "vehicle_capacity": 1, "fleet": fleet, "bidirectional": 1})
    return pd.DataFrame(rows)


def calibrate_capacity(lines: list[Line], nodes, od, edges, k_routes: int = 3,
                       horizon: float = 1.0) -> list[Line]:
    """Set every vehicle capacity so that the plan 'all lines on' carries all demand along the cheapest routes:
    capacity = largest segment load of the uncapacitated assignment / (frequency * horizon).
    Smaller plans then lose capacity, which is what makes the passenger side interesting."""
    probe = Instance("probe", nodes, od, lines, edges=edges, horizon=horizon)
    attach_routes(probe, generate_routes(probe, k_routes=k_routes))
    result = PassengerAssignment(probe, probe.default_unmet_penalty(), capacitated=False).solve(
        np.ones(probe.n_lines), np.array([line.frequency for line in lines]))
    calibrated = []
    for i, line in enumerate(lines):
        peak = result.segment_load[probe.seg_line == i].max() if np.any(probe.seg_line == i) else 0.0
        kappa = max(10.0, float(np.ceil(peak / (line.frequency * horizon))))
        calibrated.append(replace(line, vehicle_capacity=kappa))
    return calibrated


def _write_params(folder: Path, lines: list[Line], budget_share: float, **extra) -> None:
    params = {"horizon": 1.0, "lambda_op": 0.5, "lambda_pass": 0.5, "unmet_penalty": None,
              "k_routes": 3, "max_transfers": 2, "detour_factor": 2.5,
              "budget": round(budget_share * sum(l.cost for l in lines), 1),
              "fleet_total": int(np.ceil(budget_share * sum(l.fleet for l in lines)))}
    params.update(extra)
    (folder / "params.json").write_text(json.dumps(params, indent=2))


# ---------------------------------------------------------------- synthetic study
def generate_synthetic(folder: Path, seed: int = 5, rows: int = 3, cols: int = 4, n_lines: int = 10,
                       budget_share: float = 0.6) -> None:
    """Grid city of rows x cols stops, link times 2-12 min, 8 busy stops, two-way candidate lines."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    nodes = pd.DataFrame([{"id": r * cols + c, "x": float(c), "y": float(r)}
                          for r in range(rows) for c in range(cols)])
    links = [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)]
    links += [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)]
    edges = pd.DataFrame({"u": [a for a, _ in links], "v": [b for _, b in links],
                          "time": rng.integers(2, 13, len(links)).astype(float)})
    busy = sorted(rng.choice(len(nodes), size=8, replace=False))
    od = pd.DataFrame([{"origin": a, "dest": b, "demand": float(rng.integers(20, 201))}
                       for a in busy for b in busy if a != b])
    nodes.to_csv(folder / _FILES["synthetic"][0], index=False)
    edges.to_csv(folder / _FILES["synthetic"][1], index=False)
    od.to_csv(folder / _FILES["synthetic"][2], index=False)

    _generate_lines_for(folder, "synthetic", n_lines, min_stops=3, max_stops=6, budget_share=budget_share)


def generate_mandl_lines(folder: Path, n_lines: int = 8, budget_share: float = 0.6) -> None:
    """Candidate lines for the Mandl network (its nodes, links and demand are already in the folder)."""
    _generate_lines_for(Path(folder), "mandl", n_lines, min_stops=3, max_stops=8, budget_share=budget_share)


def _generate_lines_for(folder: Path, name: str, n_lines: int, min_stops: int, max_stops: int,
                        budget_share: float) -> None:
    nodes, edges, od = _read_tables(folder, _FILES[name])
    graph = build_graph(nodes, edges)
    sequences = generate_candidate_lines(graph, od, n_lines, min_stops, max_stops, bidirectional=True)
    lines = attach_line_parameters(sequences, graph, make_line_table(sequences, graph))
    lines = calibrate_capacity(lines, nodes, od, edges, horizon=read_params(folder)["horizon"])
    save_lines(lines, folder / "candidate_lines.csv")
    _write_params(folder, lines, budget_share)


# ---------------------------------------------------------------- Netherlands (your fake-data generator)
def import_network(folder: Path, n_lines: int = 10, time_scale: float = 0.01, budget_share: float = 0.6) -> None:
    """>>> NEW. Convert data/netherlands/source/{graph.csv, demand.csv} (output of your fake-data generator)
    into the project format and add candidate lines.
    Their 'distance' (100-2000, no unit given) becomes ride time in minutes: time = distance * time_scale,
    so 100-2000 gives 1-20 minutes. Their demand (10-20 per OD pair) is read as passengers per hour."""
    folder = Path(folder)
    graph = pd.read_csv(folder / "source" / "graph.csv")
    demand = pd.read_csv(folder / "source" / "demand.csv")
    nodes = pd.concat([graph[["origin_id", "origin_name"]].set_axis(["id", "name"], axis=1),
                       graph[["dest_id", "dest_name"]].set_axis(["id", "name"], axis=1)]).drop_duplicates().sort_values("id")
    edges = pd.DataFrame({"u": graph["origin_id"], "v": graph["dest_id"],
                          "time": (graph["distance"] * time_scale).round(2)})
    od = demand.rename(columns={"origin_id": "origin", "dest_id": "dest"})[["origin", "dest", "demand"]]
    names = _FILES["netherlands"]
    nodes.to_csv(folder / names[0], index=False)
    edges.to_csv(folder / names[1], index=False)
    od.to_csv(folder / names[2], index=False)
    _generate_lines_for(folder, "netherlands", n_lines, min_stops=3, max_stops=6, budget_share=budget_share)
    # The graph is sparse (chains of small towns), so passengers need more transfers than in the other cases.
    # With 2 transfers only 276 of 380 OD pairs had any route; with 4 transfers and detour factor 3, 364 have one.
    params = json.loads((folder / "params.json").read_text())
    params.update({"max_transfers": 4, "detour_factor": 3.0})
    (folder / "params.json").write_text(json.dumps(params, indent=2))
