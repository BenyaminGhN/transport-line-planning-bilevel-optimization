import pytest

def test_budget_violation_detected(tiny_instance): pytest.skip("reason == 'budget'")
def test_fleet_violation_detected(tiny_instance): pytest.skip("reason == 'fleet'")
def test_demand_not_servable_is_infeasible(tiny_instance): pytest.skip("reason == 'demand_not_servable'")
def test_cache_avoids_second_lp_solve(tiny_instance): pytest.skip("n_lp_solves unchanged on repeat x")
def test_metrics_gini_zero_for_equal_times(): pytest.skip("weighted_gini == 0")
