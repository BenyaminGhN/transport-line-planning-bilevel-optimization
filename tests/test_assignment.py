import pytest

def test_demand_conservation(tiny_instance): pytest.skip("sum_r y_kr == D_k for every k")
def test_inactive_line_gets_zero_flow(tiny_instance): pytest.skip("x_l=0 -> routes using l have y=0")
def test_capacity_binds_and_spills(tiny_instance): pytest.skip("shortest route saturated -> excess goes to route 2")
def test_uncapacitated_all_on_shortest(tiny_instance): pytest.skip("capacitated=False -> all demand on min-cost route")
def test_infeasible_when_capacity_too_small(tiny_instance): pytest.skip("AssignmentResult.feasible is False")
def test_tie_breaking_deterministic(tiny_instance): pytest.skip("same x -> identical flows across calls")
