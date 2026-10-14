import numpy as np
import pandas as pd
import pytest

from ptlp.evaluate import Evaluator
from ptlp.exact import enumerate_all
from ptlp.instance import Instance, Line, Route
from ptlp.metrics import passenger_travel_times, weighted_gini, weighted_percentile


def planning_instance(**overrides):
    values = dict(
        name="planning",
        nodes=pd.DataFrame({"id": [0, 1]}),
        od=pd.DataFrame({"k": [7], "origin": [0], "dest": [1], "demand": [100.0]}),
        lines=[
            Line(10, (0, 1), (10.0,), 6, 3, 20, 2),
            Line(20, (0, 1), (15.0,), 2, 2, 20, 1),
        ],
        routes=[
            Route(100, 7, ((10, 0),), 10.0),
            Route(200, 7, ((20, 0),), 15.0),
        ],
        budget=8,
        fleet_total=3,
        lambda_op=0.5,
        lambda_pass=0.5,
    )
    values.update(overrides)
    return Instance(**values)


def test_metrics_for_served_passengers():
    instance = planning_instance()
    instance.build()
    tt, weights = passenger_travel_times(instance, np.array([50.0, 50.0]))
    np.testing.assert_allclose(tt, [10, 15])
    assert weighted_gini(tt, weights) == pytest.approx(0.1)
    assert weighted_percentile(tt, weights, 95) == 15
    assert weighted_gini(np.array([7, 7]), np.array([2, 3])) == 0


def test_evaluator_checks_constraints_and_caches():
    evaluator = Evaluator(planning_instance(budget=6, fleet_total=2), unmet_penalty=100)
    invalid = evaluator.evaluate([1, 1])
    assert not invalid.feasible and invalid.reason == "budget"
    first = evaluator.evaluate([1, 0])
    assert first.feasible and first.assignment is not None
    assert first.assignment.unserved.sum() == pytest.approx(40)
    assert evaluator.n_lp_solves == 1
    second = evaluator.evaluate(np.array([1, 0]))
    assert second is first
    assert evaluator.n_lp_solves == 1


def test_fleet_violation_is_reported():
    evaluator = Evaluator(planning_instance(budget=100, fleet_total=2), unmet_penalty=100)
    assert evaluator.check_plan([1, 1]) == (False, "fleet")
    with pytest.raises(ValueError, match="binary"):
        evaluator.evaluate([0.5, 1])


def test_exact_enumeration_returns_all_plans_and_best():
    evaluator = Evaluator(planning_instance(), unmet_penalty=100)
    result = enumerate_all(evaluator)
    assert len(result.table) == 4
    assert result.table["bits"].tolist() == ["00", "01", "10", "11"]
    np.testing.assert_array_equal(result.best.x, [1, 1])
    assert result.best.assignment.fully_served


def test_exact_limit_is_enforced():
    evaluator = Evaluator(planning_instance(), unmet_penalty=100)
    with pytest.raises(ValueError, match="limited"):
        enumerate_all(evaluator, max_lines=1)
