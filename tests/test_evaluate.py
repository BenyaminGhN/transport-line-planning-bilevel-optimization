"""Evaluator checks.   >>> CHANGED: placeholders replaced (the soft model has no 'demand_not_servable') <<<"""
import numpy as np
import pytest

from ptlp.evaluate import Evaluator
from ptlp.metrics import weighted_gini
from tests.helpers import two_line_instance


def test_budget_violation_detected():
    evaluator = Evaluator(two_line_instance(budget=7), unmet_penalty=100)
    result = evaluator.evaluate([1, 1])             # costs 8 > 7
    assert not result.feasible and result.reason == "budget" and result.violation > 0


def test_fleet_violation_detected():
    result = Evaluator(two_line_instance(fleet_total=1), unmet_penalty=100).evaluate([1, 1])
    assert not result.feasible and result.reason == "fleet"


def test_unservable_demand_is_penalised_not_infeasible():
    result = Evaluator(two_line_instance(), unmet_penalty=100).evaluate([0, 0])
    assert result.feasible and result.served_fraction == 0 and result.unserved == pytest.approx(100)


def test_infeasible_fitness_is_worse_than_every_feasible_plan():
    evaluator = Evaluator(two_line_instance(budget=7), unmet_penalty=100)
    worst_feasible = max(evaluator.fitness(evaluator.evaluate(x)) for x in ([0, 0], [1, 0], [0, 1]))
    assert evaluator.fitness(evaluator.evaluate([1, 1])) > worst_feasible


def test_cache_avoids_second_lp_solve():
    evaluator = Evaluator(two_line_instance(), unmet_penalty=100)
    evaluator.evaluate([1, 0])
    solves = evaluator.n_lp_solves
    evaluator.evaluate([1, 0])
    assert evaluator.n_lp_solves == solves


def test_objective_is_weighted_normalised_sum():
    instance = two_line_instance()
    evaluator = Evaluator(instance, unmet_penalty=100)
    r = evaluator.evaluate([1, 1])
    expected = 0.5 * 8 / evaluator.c_op_ref + 0.5 * (60 * 10 + 40 * 15) / evaluator.c_pass_ref
    assert r.objective == pytest.approx(expected)
    assert (r.c_op_norm, r.c_pass_norm) == pytest.approx((8 / evaluator.c_op_ref, 1200 / evaluator.c_pass_ref))


def test_references_use_the_same_omega_and_do_not_leak_between_evaluators():
    instance = two_line_instance(cap_a=10)          # all-lines reference is uncapacitated, so no unserved
    a, b = Evaluator(instance, unmet_penalty=100), Evaluator(instance, unmet_penalty=1000)
    assert a.c_pass_ref == pytest.approx(b.c_pass_ref) == pytest.approx(1000.0)
    assert instance.c_pass_ref is None               # the instance itself is not modified


def test_gini_and_p95_cover_served_passengers_only():
    r = Evaluator(two_line_instance(), unmet_penalty=100).evaluate([1, 0])     # 60 served, 40 not
    assert r.gini == pytest.approx(0.0)              # everybody served rides the same 10-minute route
    assert r.p95 == pytest.approx(10.0)


def test_metrics_gini_zero_for_equal_times():
    assert weighted_gini(np.array([7.0, 7.0]), np.array([2.0, 3.0])) == 0


def test_case7_different_plans_give_different_travel_time_distributions():
    evaluator = Evaluator(two_line_instance(), unmet_penalty=100)
    assert evaluator.evaluate([1, 0]).gini == pytest.approx(0.0)       # one route used
    assert evaluator.evaluate([1, 1]).gini > 0.05                      # 10- and 15-minute rides mixed
    assert evaluator.evaluate([1, 1]).p95 == pytest.approx(15.0)
