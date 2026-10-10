"""File reading only. No generation, no optimisation."""
from __future__ import annotations

from pathlib import Path

from .instance import Instance


def load_synthetic(folder: Path, generate_lines: bool = False) -> Instance:
    """Input: data/synthetic/ (nodes.csv, edges.csv, od_demand.csv, candidate_lines.csv).
    Output: Instance with nodes, od, lines (routes still empty, build() not yet called)."""
    raise NotImplementedError


def load_mandl(folder: Path) -> Instance:
    """Input: data/mandl/ (network.txt, ODMatrix.txt, candidate_lines.csv, line_parameters.csv).
    Output: Instance (same contract as load_synthetic). Nodes have no x,y -> viz uses layout fallback."""
    raise NotImplementedError


def load_instance(name: str, root: Path = Path("data")) -> Instance:
    """Input: 'synthetic' | 'mandl'. Output: fully prepared Instance
    (lines -> routes -> build() -> set_references() already done)."""
    raise NotImplementedError
