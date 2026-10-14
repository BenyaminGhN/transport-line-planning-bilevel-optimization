"""Data files, loaders, line and route generation.   >>> NEW FILE <<<"""
import pandas as pd
import pytest

from ptlp.datagen import generate_synthetic
from ptlp.io import load_instance


def test_mandl_matches_the_benchmark_description():
    instance = load_instance("mandl")
    assert len(instance.nodes) == 15
    assert len(instance.edges) == 42                       # 21 links, both directions
    assert instance.demand.sum() == 15570
    assert instance.n_routes > 0 and all(line.bidirectional for line in instance.lines)


def test_synthetic_instance_is_complete():
    instance = load_instance("synthetic")
    assert instance.n_od == 56 and instance.n_lines >= 10
    assert set(instance.od["origin"]) == set(instance.od["dest"])
    assert instance.budget > 0 and instance.fleet_total > 0


def test_generation_is_deterministic(tmp_path):
    generate_synthetic(tmp_path / "a")
    generate_synthetic(tmp_path / "b")
    for name in ("nodes.csv", "edges.csv", "od_demand.csv", "candidate_lines.csv", "params.json"):
        assert (tmp_path / "a" / name).read_text() == (tmp_path / "b" / name).read_text()


def test_every_demand_node_is_on_a_line_and_has_a_route():
    instance = load_instance("synthetic")
    on_lines = {stop for line in instance.lines for stop in line.stops}
    assert set(instance.od["origin"]) <= on_lines
    assert len({instance.od["k"].iloc[k] for k in instance.route_od}) == instance.n_od   # every OD pair has a route


def test_unknown_data_set_is_rejected():
    with pytest.raises(ValueError):
        load_instance("nowhere")


def test_two_way_line_routes_can_ride_backwards():
    instance = load_instance("synthetic")
    assert any(s >= instance.lines[[l.id for l in instance.lines].index(lid)].n_forward
               for route in instance.routes for lid, s in route.legs)


def test_netherlands_data_is_converted_correctly():
    """>>> NEW: your fake-data generator output (20 cities, 29 links, 380 OD pairs of 10-20 passengers)."""
    instance = load_instance("netherlands")
    assert len(instance.nodes) == 20 and len(instance.edges) == 58          # 29 links, both directions
    assert instance.n_od == 380 and instance.demand.min() >= 10 and instance.demand.max() <= 20
    assert 1.0 <= instance.edges["time"].min() and instance.edges["time"].max() <= 20.0   # minutes
    assert len(set(instance.route_od)) >= 360          # (4 transfers allowed) almost every OD pair has a route
