"""End-to-end line-planning demo using a small in-memory network.

Run from the repository root:
    python -m scripts.run_line_planning
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ptlp.evaluate import Evaluator
from ptlp.exact import enumerate_all
from ptlp.ga import GAConfig, GeneticAlgorithm
from ptlp.instance import Instance
from ptlp.lines import attach_line_parameters, build_graph, generate_candidate_lines
from ptlp.routes import attach_routes, generate_routes


def example_instance() -> Instance:
    """Build a two-corridor network with one OD pair and two candidate lines."""
    nodes = pd.DataFrame({
        "id": [0, 1, 2, 3],
        "x": [0.0, 1.0, 2.0, 1.0],
        "y": [0.0, 0.0, 0.0, 1.0],
    })
    edges = pd.DataFrame({
        "u": [0, 1, 0, 3],
        "v": [1, 2, 3, 2],
        "time": [5.0, 5.0, 6.0, 6.0],
    })
    od = pd.DataFrame({
        "k": [0],
        "origin": [0],
        "dest": [2],
        "demand": [100.0],
    })

    graph = build_graph(nodes, edges)
    stop_sequences = generate_candidate_lines(
        graph, od, n_lines=2, min_stops=3, max_stops=3
    )
    parameters = pd.DataFrame({
        "id": [10, 20],
        "cost": [6.0, 2.0],
        "frequency": [3.0, 2.0],
        "vehicle_capacity": [20, 20],
        "fleet": [2, 1],
    })
    lines = attach_line_parameters(stop_sequences, graph, parameters)
    instance = Instance(
        name="line-planning-demo",
        nodes=nodes,
        od=od,
        lines=lines,
        horizon=1.0,
        budget=8.0,
        fleet_total=3,
        lambda_op=0.5,
        lambda_pass=0.5,
    )
    attach_routes(instance, generate_routes(instance, k_routes=2))
    return instance


def _bits(x: np.ndarray) -> str:
    return "".join(str(int(value)) for value in x)


def main() -> None:
    instance = example_instance()
    penalty = 100.0

    print("Candidate lines:")
    for line in instance.lines:
        print(
            f"  line {line.id}: stops={list(line.stops)}, cost={line.cost:.1f}, "
            f"frequency={line.frequency:.1f}, capacity={line.vehicle_capacity}"
        )
    print("Passenger routes:")
    for route in instance.routes:
        print(
            f"  route {route.id}: OD={route.od_index}, lines={sorted(route.lines)}, "
            f"travel_time={route.cost:.1f}"
        )

    exact_evaluator = Evaluator(instance, penalty=penalty)
    exact = enumerate_all(exact_evaluator)

    ga_evaluator = Evaluator(instance, penalty=penalty)
    config = GAConfig(pop_size=8, n_generations=6, mutation_rate=0.1, seed=4)
    ga = GeneticAlgorithm(ga_evaluator, config).run()

    exact_unserved = float(exact.best.assignment.unserved.sum())
    ga_unserved = float(ga.best.assignment.unserved.sum())
    gap = ga.best.objective - exact.best.objective

    print("Exact enumeration:")
    print(f"  plans evaluated: {len(exact.table)}")
    print(f"  best x: {_bits(exact.best.x)}")
    print(f"  objective: {exact.best.objective:.6f}")
    print(f"  unserved demand: {exact_unserved:.1f}")
    print(f"  Gini: {exact.best.gini:.6f}")
    print(f"  P95 travel time: {exact.best.p95:.1f}")

    print("Genetic algorithm:")
    print(f"  generations: {len(ga.history)}")
    print(f"  best x: {_bits(ga.best.x)}")
    print(f"  objective: {ga.best.objective:.6f}")
    print(f"  unserved demand: {ga_unserved:.1f}")
    print(f"  gap to exact: {gap:.6f}")


if __name__ == "__main__":
    main()
