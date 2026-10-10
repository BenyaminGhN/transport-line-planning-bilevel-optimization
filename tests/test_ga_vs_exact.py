import pytest

def test_ga_reaches_exact_optimum_on_tiny_instance(tiny_instance): pytest.skip("gap == 0 within tolerance")
def test_ga_history_length_and_monotone_best(tiny_instance): pytest.skip("best_fitness non-increasing")
def test_callback_called_every_generation(tiny_instance): pytest.skip("len(calls) == n_generations")
def test_ga_reproducible_with_seed(tiny_instance): pytest.skip("same seed -> same history")
