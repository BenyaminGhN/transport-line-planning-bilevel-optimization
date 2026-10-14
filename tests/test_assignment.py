"""Checks of the passenger LP, one per case of PDF section 16.1.   >>> CHANGED: placeholders replaced <<<"""
import numpy as np
import pytest

from ptlp.assignment import PassengerAssignment
from tests.helpers import flows_by_line, two_line_instance

BOTH, ONLY_A, ONLY_B, NONE = [1, 1], [1, 0], [0, 1], [0, 0]
FREQ = [1.0, 1.0]


def solve(instance, x, omega=100.0, **kwargs):
    return PassengerAssignment(instance, omega, **kwargs).solve(x, FREQ)


def test_case1_fastest_route_is_full_so_rest_takes_slower_route(tiny_instance):
    result = solve(tiny_instance, BOTH)
    assert flows_by_line(tiny_instance, result) == {1: 60.0, 2: 40.0}
    assert result.fully_served
    assert result.c_pass == pytest.approx(60 * 10 + 40 * 15)


def test_case2_alternative_route_is_used_when_the_fast_line_is_off(tiny_instance):
    result = solve(tiny_instance, ONLY_B)
    assert flows_by_line(tiny_instance, result) == {1: 0.0, 2: 100.0}


def test_case3_extra_line_adds_capacity(tiny_instance):
    only_a, both = solve(tiny_instance, ONLY_A), solve(tiny_instance, BOTH)
    assert only_a.served_fraction == pytest.approx(0.6)
    assert both.served_fraction == pytest.approx(1.0)


def test_case4_cheap_plan_leaves_demand_unserved(tiny_instance):
    result = solve(tiny_instance, ONLY_A)
    assert not result.fully_served
    assert result.unserved_total == pytest.approx(40)
    assert result.penalty_cost == pytest.approx(100 * 40)
    assert result.objective == pytest.approx(60 * 10 + 100 * 40)


def test_case5_same_times_different_capacity_gives_different_service():
    small = solve(two_line_instance(cap_a=30, cap_b=30, time_a=10, time_b=10), BOTH)
    large = solve(two_line_instance(cap_a=50, cap_b=50, time_a=10, time_b=10), BOTH)
    assert small.served_fraction == pytest.approx(0.6)
    assert large.served_fraction == pytest.approx(1.0)


def test_case6_travel_times_decide_who_uses_which_route():
    fast_a = two_line_instance(cap_a=200, cap_b=200, time_a=10, time_b=15)
    fast_b = two_line_instance(cap_a=200, cap_b=200, time_a=15, time_b=10)
    assert flows_by_line(fast_a, solve(fast_a, BOTH)) == {1: 100.0, 2: 0.0}
    assert flows_by_line(fast_b, solve(fast_b, BOTH)) == {1: 0.0, 2: 100.0}


def test_demand_is_conserved(tiny_instance):
    for x in (BOTH, ONLY_A, ONLY_B, NONE):
        result = solve(tiny_instance, x)
        assert tiny_instance.A_od @ result.flows + result.unserved == pytest.approx(tiny_instance.demand)


def test_inactive_line_gets_no_flow(tiny_instance):
    assert flows_by_line(tiny_instance, solve(tiny_instance, ONLY_B))[1] == 0.0


def test_segment_load_never_exceeds_capacity(tiny_instance):
    result = solve(tiny_instance, BOTH)
    assert np.all(result.segment_load <= tiny_instance.seg_capacity + 1e-7)
    assert result.segment_util[0] == pytest.approx(1.0)       # line A is full


def test_uncapacitated_sends_everybody_on_the_fastest_route(tiny_instance):
    result = solve(tiny_instance, BOTH, capacitated=False)
    assert flows_by_line(tiny_instance, result) == {1: 100.0, 2: 0.0}


def test_empty_plan_is_always_feasible_and_all_unserved(tiny_instance):
    result = solve(tiny_instance, NONE)
    assert result.feasible and result.served_fraction == 0.0


def test_tie_breaking_prefers_the_route_listed_first():
    """Two identical routes: the first listed gets the passengers, whatever the order is."""
    for order, winner in ((("A", "B"), 1), (("B", "A"), 2)):
        instance = two_line_instance(time_a=10, time_b=10, order=order)
        result = solve(instance, BOTH)
        assert flows_by_line(instance, result)[winner] == 60.0 or flows_by_line(instance, result)[winner] == 100.0
        assert flows_by_line(instance, result)[winner] == max(flows_by_line(instance, result).values())
        assert np.allclose(result.flows, solve(instance, BOTH).flows)      # repeatable
