"""Genetic algorithm for the leader. Emits one GenerationRecord per generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np

from .evaluate import Evaluator, EvaluationResult


@dataclass
class GAConfig:
    pop_size: int = 40
    n_generations: int = 60
    crossover_rate: float = 0.9
    mutation_rate: float = 0.02      # per bit
    tournament_k: int = 3
    elitism: int = 2
    seed: int = 0
    patience: int | None = None      # early stop if no improvement


@dataclass
class GenerationRecord:
    """Snapshot after each generation. This is all the visualisation needs."""
    generation: int
    population: np.ndarray           # (pop, n_lines) 0/1
    fitness: np.ndarray              # (pop,)
    c_op: np.ndarray                 # (pop,) normalised
    c_pass: np.ndarray               # (pop,) normalised (nan if infeasible)
    feasible: np.ndarray             # (pop,) bool
    best_x: np.ndarray               # best-so-far plan
    best_fitness: float
    best_result: EvaluationResult    # includes flows -> network plot without re-solving
    diversity: float                 # mean pairwise Hamming distance / n_lines
    n_lp_solves: int                 # cumulative
    elapsed_s: float


@dataclass
class GAResult:
    best: EvaluationResult
    history: list[GenerationRecord] = field(default_factory=list)
    runtime_s: float = 0.0


Callback = Callable[[GenerationRecord], None]


class GeneticAlgorithm:
    def __init__(self, evaluator: Evaluator, config: GAConfig,
                 on_generation: Callback | None = None):
        """on_generation is called after EVERY generation (hook for live plots / logging)."""
        if config.pop_size < 2:
            raise ValueError("pop_size must be at least 2")
        if config.n_generations < 1:
            raise ValueError("n_generations must be positive")
        if not 0 <= config.crossover_rate <= 1 or not 0 <= config.mutation_rate <= 1:
            raise ValueError("crossover_rate and mutation_rate must lie in [0, 1]")
        if not 1 <= config.tournament_k <= config.pop_size:
            raise ValueError("tournament_k must lie between 1 and pop_size")
        if not 0 <= config.elitism < config.pop_size:
            raise ValueError("elitism must lie between 0 and pop_size - 1")
        if config.patience is not None and config.patience < 1:
            raise ValueError("patience must be positive when provided")
        self.evaluator = evaluator
        self.config = config
        self.on_generation = on_generation
        self.rng = np.random.default_rng(config.seed)
        self._best_result: EvaluationResult | None = None
        self._best_fitness = np.inf
        self._started_at = 0.0

    def run(self) -> GAResult:
        """Output: GAResult (best plan + full history)."""
        self.rng = np.random.default_rng(self.config.seed)
        self._best_result = None
        self._best_fitness = np.inf
        self._started_at = perf_counter()
        history: list[GenerationRecord] = []
        population = self._init_population()
        generations_without_improvement = 0

        for generation in range(self.config.n_generations):
            results = self._evaluate_population(population)
            previous_best = self._best_fitness
            record = self._make_record(generation, population, results)
            history.append(record)
            if self.on_generation is not None:
                self.on_generation(record)
            generations_without_improvement = (
                0 if self._best_fitness < previous_best - 1e-12
                else generations_without_improvement + 1
            )
            if (self.config.patience is not None
                    and generations_without_improvement >= self.config.patience):
                break
            if generation == self.config.n_generations - 1:
                break

            fitness = np.asarray([self.evaluator.fitness(result) for result in results])
            elite_indices = np.argsort(fitness, kind="stable")[:self.config.elitism]
            next_population = [population[index].copy() for index in elite_indices]
            parents = self._select(population, fitness)
            parent_index = 0
            while len(next_population) < self.config.pop_size:
                p1 = parents[parent_index % len(parents)]
                p2 = parents[(parent_index + 1) % len(parents)]
                parent_index += 2
                child1, child2 = self._crossover(p1, p2)
                next_population.append(self._repair(self._mutate(child1)))
                if len(next_population) < self.config.pop_size:
                    next_population.append(self._repair(self._mutate(child2)))
            population = np.asarray(next_population, dtype=np.uint8)

        if self._best_result is None:
            raise RuntimeError("genetic algorithm did not evaluate a population")
        return GAResult(self._best_result, history, perf_counter() - self._started_at)

    def _init_population(self) -> np.ndarray:
        """Output (pop, n_lines). Random bits, ideally biased to pass budget/fleet."""
        shape = (self.config.pop_size, self.evaluator.instance.n_lines)
        population = self.rng.integers(0, 2, size=shape, dtype=np.uint8)
        population[0] = 0
        if self.config.pop_size > 1:
            population[1] = 1
        return np.asarray([self._repair(individual) for individual in population], dtype=np.uint8)

    def _evaluate_population(self, pop: np.ndarray) -> list[EvaluationResult]:
        return [self.evaluator.evaluate(individual) for individual in pop]

    def _repair(self, x: np.ndarray) -> np.ndarray:
        """Drop lines (e.g. worst cost-per-benefit) until budget and fleet hold."""
        x = np.asarray(x, dtype=np.uint8).copy()
        instance = self.evaluator.instance
        cost_scale = max(float(np.max(instance.line_cost, initial=0)), 1.0)
        fleet_scale = max(float(np.max(instance.line_fleet, initial=0)), 1.0)
        scores = instance.line_cost / cost_scale + instance.line_fleet / fleet_scale
        while not self.evaluator.check_plan(x)[0]:
            active = np.flatnonzero(x)
            if not len(active):
                break
            worst = active[np.argmax(scores[active])]
            x[worst] = 0
        return x

    def _select(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Tournament selection. Output: parents (pop, n_lines)."""
        parents = []
        for _ in range(self.config.pop_size):
            candidates = self.rng.choice(len(pop), size=self.config.tournament_k, replace=False)
            winner = candidates[np.argmin(fitness[candidates])]
            parents.append(pop[winner].copy())
        return np.asarray(parents, dtype=np.uint8)

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p1, p2 = p1.copy(), p2.copy()
        if len(p1) < 2 or self.rng.random() >= self.config.crossover_rate:
            return p1, p2
        point = int(self.rng.integers(1, len(p1)))
        return (np.concatenate((p1[:point], p2[point:])),
                np.concatenate((p2[:point], p1[point:])))

    def _mutate(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        flips = self.rng.random(len(x)) < self.config.mutation_rate
        x[flips] = 1 - x[flips]
        return x

    def _make_record(self, gen: int, pop: np.ndarray,
                     results: list[EvaluationResult]) -> GenerationRecord:
        fitness = np.asarray([self.evaluator.fitness(result) for result in results], dtype=float)
        generation_best_index = int(np.argmin(fitness))
        if fitness[generation_best_index] < self._best_fitness:
            self._best_fitness = float(fitness[generation_best_index])
            self._best_result = results[generation_best_index]
        assert self._best_result is not None

        feasible = np.asarray([result.feasible for result in results], dtype=bool)
        c_op = np.asarray([
            result.c_op / self.evaluator.instance.c_op_ref for result in results
        ], dtype=float)
        c_pass = np.asarray([
            result.c_pass / self.evaluator.instance.c_pass_ref if result.feasible else np.nan
            for result in results
        ], dtype=float)
        n_people, n_lines = pop.shape
        if n_people < 2 or n_lines == 0:
            diversity = 0.0
        else:
            ones = pop.sum(axis=0)
            differing_pairs = float(np.sum(ones * (n_people - ones)))
            diversity = differing_pairs / ((n_people * (n_people - 1) / 2) * n_lines)
        return GenerationRecord(
            generation=gen,
            population=pop.copy(),
            fitness=fitness,
            c_op=c_op,
            c_pass=c_pass,
            feasible=feasible,
            best_x=self._best_result.x.copy(),
            best_fitness=self._best_fitness,
            best_result=self._best_result,
            diversity=float(diversity),
            n_lp_solves=self.evaluator.n_lp_solves,
            elapsed_s=perf_counter() - self._started_at,
        )
