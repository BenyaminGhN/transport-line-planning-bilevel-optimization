import numpy as np
import pandas as pd

from ptlp.evaluate import Evaluator
from ptlp.exact import enumerate_all
from ptlp.ga import GAConfig, GeneticAlgorithm
from ptlp.instance import Instance, Line, Route


def planning_instance():
    return Instance(
        name="ga-planning",
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
    )


def run_ga(seed=4, callback=None):
    evaluator = Evaluator(planning_instance(), penalty=100)
    config = GAConfig(pop_size=8, n_generations=6, mutation_rate=0.1, seed=seed)
    return GeneticAlgorithm(evaluator, config, on_generation=callback).run()


def test_ga_matches_exact_on_small_instance():
    exact = enumerate_all(Evaluator(planning_instance(), penalty=100))
    result = run_ga()
    assert result.best.objective == exact.best.objective
    np.testing.assert_array_equal(result.best.x, exact.best.x)


def test_history_is_monotone_and_callback_runs_each_generation():
    calls = []
    result = run_ga(callback=calls.append)
    assert len(result.history) == len(calls) == 6
    best = np.asarray([record.best_fitness for record in result.history])
    assert np.all(np.diff(best) <= 1e-12)
    assert all(0 <= record.diversity <= 1 for record in result.history)


def test_ga_is_reproducible():
    first, second = run_ga(seed=9), run_ga(seed=9)
    np.testing.assert_array_equal(first.best.x, second.best.x)
    np.testing.assert_allclose(
        [record.best_fitness for record in first.history],
        [record.best_fitness for record in second.history],
    )
    for left, right in zip(first.history, second.history):
        np.testing.assert_array_equal(left.population, right.population)
