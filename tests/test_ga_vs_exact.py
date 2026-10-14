"""GA behaviour on a small real instance (the synthetic network).   >>> CHANGED: placeholders replaced <<<"""
import numpy as np
import pytest

from ptlp.evaluate import Evaluator
from ptlp.exact import enumerate_all
from ptlp.ga import GAConfig, GeneticAlgorithm
from ptlp.io import load_instance


@pytest.fixture(scope="module")
def synthetic():
    return load_instance("synthetic")


@pytest.fixture(scope="module")
def exact(synthetic):
    return enumerate_all(Evaluator(synthetic))


def run_ga(instance, seed=0, callback=None, generations=20):
    return GeneticAlgorithm(Evaluator(instance), GAConfig(pop_size=20, n_generations=generations, seed=seed),
                            callback).run()


def test_ga_reaches_exact_optimum(synthetic, exact):
    assert run_ga(synthetic).best.objective == pytest.approx(exact.best.objective)


def test_ga_never_beats_the_exact_optimum(synthetic, exact):
    for seed in range(3):
        assert run_ga(synthetic, seed, generations=3).best.objective >= exact.best.objective - 1e-9


def test_history_length_and_monotone_best(synthetic):
    result = run_ga(synthetic, generations=8)
    best = [r.best_fitness for r in result.history]
    assert len(best) == 8 and all(b <= a + 1e-12 for a, b in zip(best, best[1:]))


def test_callback_called_every_generation(synthetic):
    calls = []
    run_ga(synthetic, callback=calls.append, generations=5)
    assert [r.generation for r in calls] == [0, 1, 2, 3, 4]


def test_ga_reproducible_with_seed(synthetic):
    a, b = run_ga(synthetic, seed=3, generations=6), run_ga(synthetic, seed=3, generations=6)
    assert [r.best_fitness for r in a.history] == [r.best_fitness for r in b.history]


def test_every_individual_respects_budget_and_fleet(synthetic):
    result = run_ga(synthetic, generations=6)
    for record in result.history:
        assert record.feasible.all()
        assert np.all(record.population @ synthetic.line_cost <= synthetic.budget + 1e-9)
        assert np.all(record.population @ synthetic.line_fleet <= synthetic.fleet_total + 1e-9)


def test_lp_solves_are_counted_per_run(synthetic):
    evaluator = Evaluator(synthetic)
    first = GeneticAlgorithm(evaluator, GAConfig(pop_size=10, n_generations=3)).run()
    second = GeneticAlgorithm(evaluator, GAConfig(pop_size=10, n_generations=3, seed=1)).run()
    assert first.history[0].n_lp_solves > 0 and second.n_lp_solves <= evaluator.n_lp_solves - first.n_lp_solves
