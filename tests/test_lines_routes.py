import networkx as nx
import pandas as pd
import pytest

from ptlp.instance import Instance
from ptlp.lines import (
    attach_line_parameters,
    build_graph,
    generate_candidate_lines,
    load_lines,
    save_lines,
)
from ptlp.routes import attach_routes, build_line_layered_graph, generate_routes


@pytest.fixture
def network_tables():
    nodes = pd.DataFrame({"id": [0, 1, 2, 3], "x": [0, 1, 2, 1], "y": [0, 0, 0, 1]})
    edges = pd.DataFrame({
        "u": [0, 1, 0, 3, 3, 2],
        "v": [1, 2, 3, 2, 1, 0],
        "time": [2, 2, 3, 3, 1, 4],
    })
    od = pd.DataFrame({"k": [10, 20], "origin": [0, 3], "dest": [2, 2], "demand": [100, 20]})
    return nodes, edges, od


def test_build_graph_and_candidate_ranking(network_tables):
    nodes, edges, od = network_tables
    graph = build_graph(nodes, edges)
    lines = generate_candidate_lines(graph, od, n_lines=3, min_stops=2, max_stops=4)
    assert lines[0] == [0, 1, 2]
    assert [0, 3, 2] in lines
    assert len(lines) == len({tuple(line) for line in lines}) == 3
    assert graph[0][1]["time"] == 2


def test_graph_rejects_invalid_edges(network_tables):
    nodes, edges, _ = network_tables
    bad = pd.concat([edges, pd.DataFrame({"u": [9], "v": [0], "time": [1]})], ignore_index=True)
    with pytest.raises(ValueError, match="known node"):
        build_graph(nodes, bad)


def test_attach_and_round_trip_lines(network_tables, tmp_path):
    nodes, edges, _ = network_tables
    graph = build_graph(nodes, edges)
    params = pd.DataFrame({
        "id": [10, 20], "cost": [7, 9], "frequency": [3, 2],
        "vehicle_capacity": [40, 30], "fleet": [2, 1],
    })
    lines = attach_line_parameters([[0, 1, 2], [0, 3, 2]], graph, params)
    assert lines[0].segment_times == (2.0, 2.0)
    path = tmp_path / "candidate_lines.csv"
    save_lines(lines, path)
    assert load_lines(path) == lines


def test_layered_graph_and_transfer_route(network_tables):
    nodes, edges, od = network_tables
    graph = build_graph(nodes, edges)
    params = pd.DataFrame({
        "id": [10, 20], "cost": [7, 9], "frequency": [3, 2],
        "vehicle_capacity": [40, 30], "fleet": [2, 1],
    })
    lines = attach_line_parameters([[0, 1], [1, 2]], graph, params)
    instance = Instance("transfer", nodes, od.iloc[[0]].copy(), lines)
    layered = build_line_layered_graph(lines)
    assert layered[(1, 10)][(1, 20)]["kind"] == "transfer"
    routes = generate_routes(instance, k_routes=2)
    assert len(routes) == 1
    assert routes[0].legs == ((10, 0), (20, 0))
    assert routes[0].cost == pytest.approx(4)
    attach_routes(instance, routes)
    assert instance.n_routes == 1
    assert instance.A_cap.shape == (2, 1)


def test_disconnected_od_has_no_route(network_tables):
    nodes, edges, _ = network_tables
    graph = build_graph(nodes, edges)
    params = pd.DataFrame({
        "id": [10], "cost": [7], "frequency": [3], "vehicle_capacity": [40], "fleet": [2],
    })
    lines = attach_line_parameters([[0, 1]], graph, params)
    od = pd.DataFrame({"k": [99], "origin": [0], "dest": [2], "demand": [10]})
    assert generate_routes(Instance("disconnected", nodes, od, lines)) == []


def test_candidate_generation_rejects_bad_limits(network_tables):
    nodes, edges, od = network_tables
    graph = build_graph(nodes, edges)
    with pytest.raises(ValueError):
        generate_candidate_lines(graph, od, n_lines=1, min_stops=1)
