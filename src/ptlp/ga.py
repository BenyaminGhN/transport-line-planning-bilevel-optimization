"""Genetic algorithm for the leader. Emits one GenerationRecord per generation."""
from __future__ import annotations

from dataclasses import dataclass, field
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
        raise NotImplementedError

    def run(self) -> GAResult:
        """Output: GAResult (best plan + full history)."""
        raise NotImplementedError

    def _init_population(self) -> np.ndarray:
        """Output (pop, n_lines). Random bits, ideally biased to pass budget/fleet."""
        raise NotImplementedError

    def _evaluate_population(self, pop: np.ndarray) -> list[EvaluationResult]: raise NotImplementedError
    def _repair(self, x: np.ndarray) -> np.ndarray:
        """Drop lines (e.g. worst cost-per-benefit) until budget and fleet hold."""
        raise NotImplementedError
    def _select(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Tournament selection. Output: parents (pop, n_lines)."""
        raise NotImplementedError
    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]: raise NotImplementedError
    def _mutate(self, x: np.ndarray) -> np.ndarray: raise NotImplementedError
    def _make_record(self, gen: int, pop: np.ndarray, results: list[EvaluationResult]) -> GenerationRecord: raise NotImplementedError
