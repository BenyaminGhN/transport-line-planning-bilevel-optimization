import numpy as np
import pandas as pd
import pytest

from ptlp.assignment import PassengerAssignment
from ptlp.instance import Instance, Line, Route
from scripts.run_assignment import example_instance


def test_capacity_shortage_is_penalized():
    inst = example_instance()
    result = PassengerAssignment(inst, 100).solve([1, 1], [3, 1])
    assert result.feasible and not result.fully_served
    np.testing.assert_allclose(result.flows, [60, 20])
    np.testing.assert_allclose(result.unserved, [20])
    np.testing.assert_allclose(inst.A_od @ result.flows + result.unserved, inst.demand)
    assert result.c_pass == pytest.approx(900)
    assert result.penalty_cost == pytest.approx(2000)
    assert result.objective == pytest.approx(2900)


def test_frequency_change_reuses_solver():
    solver = PassengerAssignment(example_instance(), 100)
    solver.solve([1, 1], [3, 1])
    result = solver.solve([1, 1], [5, 1])
    np.testing.assert_allclose(result.flows, [100, 0])
    assert result.fully_served
    assert result.objective == pytest.approx(1000)


@pytest.mark.parametrize("x,f", [([0, 0], [5, 5]), ([1, 1], [0, 0])])
def test_no_service_is_all_unserved(x, f):
    result = PassengerAssignment(example_instance(), 100).solve(x, f)
    assert result.feasible
    np.testing.assert_allclose(result.flows, [0, 0])
    assert result.objective == pytest.approx(10000)
    assert np.all(np.isnan(result.segment_util))


def test_x_disables_line_with_positive_frequency():
    result = PassengerAssignment(example_instance(), 100).solve([0, 1], [5, 5])
    np.testing.assert_allclose(result.flows, [0, 100])
    assert result.objective == pytest.approx(1500)


def test_low_penalty_can_prefer_unserved_even_with_capacity():
    result = PassengerAssignment(example_instance(), 5).solve([1, 1], [5, 5])
    np.testing.assert_allclose(result.unserved, [100])
    assert result.objective == pytest.approx(500)


def test_equal_penalty_and_time_accepts_any_optimum():
    result = PassengerAssignment(example_instance(), 10).solve([1, 1], [5, 5])
    assert result.feasible
    assert result.objective == pytest.approx(1000)
    assert result.flows.sum() + result.unserved.sum() == pytest.approx(100)


def test_od_without_routes():
    inst = example_instance()
    inst.routes = []
    result = PassengerAssignment(inst, 100).solve([1, 1], [5, 5])
    assert result.feasible and result.flows.size == 0
    np.testing.assert_allclose(result.unserved, [100])


def test_uncapacitated_baseline():
    result = PassengerAssignment(example_instance(), 100, capacitated=False).solve([1, 1], [1, 1])
    np.testing.assert_allclose(result.flows, [100, 0])
    assert result.fully_served


def test_shared_capacity_across_od_and_transfer_availability():
    inst = Instance(
        "shared", pd.DataFrame({"id": [0, 1, 2]}),
        pd.DataFrame({"k": [8, 9], "origin": [0, 0], "dest": [1, 2], "demand": [40, 40]}),
        [Line(10, (0, 1), (10,), 0, 1, 50, 1),
         Line(20, (1, 2), (5,), 0, 1, 50, 1)],
        [Route(1, 8, ((10, 0),), 10), Route(2, 9, ((10, 0), (20, 0)), 15)],
    )
    solver = PassengerAssignment(inst, 100)
    result = solver.solve([1, 1], [1, 1])
    np.testing.assert_allclose(result.flows, [40, 10])
    np.testing.assert_allclose(result.segment_load, [50, 10])
    np.testing.assert_allclose(result.unserved, [0, 30])
    result = solver.solve([1, 0], [1, 1])
    np.testing.assert_allclose(result.flows, [40, 0])


@pytest.mark.parametrize("x,f", [([2, 1], [1, 1]), ([1], [1]),
                                      ([1, 1], [-1, 1]), ([1, 1], [np.nan, 1])])
def test_bad_plan_rejected(x, f):
    with pytest.raises(ValueError):
        PassengerAssignment(example_instance(), 100).solve(x, f)


def test_invalid_route_cost_rejected():
    inst = example_instance()
    inst.routes = [Route(1, 7, ((10, 0),), 99)]
    with pytest.raises(ValueError, match="Route cost"):
        PassengerAssignment(inst, 100)
