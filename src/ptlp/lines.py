"""Candidate-line generation. Run once per instance, then save to candidate_lines.csv."""
from __future__ import annotations

import networkx as nx
import pandas as pd

from .instance import Line


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """Input: nodes (id,x,y), edges (u,v,time). Output: directed graph, edge attr 'time'."""
    raise NotImplementedError


def generate_candidate_lines(graph: nx.DiGraph, od: pd.DataFrame, n_lines: int,
                             min_stops: int = 3, max_stops: int = 8, seed: int = 0) -> list[list[int]]:
    """Input: graph, OD table, number of lines wanted.
    Output: list of stop sequences (e.g. shortest paths between high-demand OD pairs)."""
    raise NotImplementedError


def attach_line_parameters(stop_sequences: list[list[int]], graph: nx.DiGraph,
                           params: pd.DataFrame) -> list[Line]:
    """Input: stop sequences + table (cost, frequency, vehicle_capacity, fleet).
    Output: list[Line] with segment_times taken from the graph."""
    raise NotImplementedError


def save_lines(lines: list[Line], path) -> None: raise NotImplementedError
def load_lines(path) -> list[Line]: raise NotImplementedError
