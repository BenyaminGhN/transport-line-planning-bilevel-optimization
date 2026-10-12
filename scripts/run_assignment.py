"""Run from the repository root: python -m scripts.run_assignment"""
import numpy as np
import pandas as pd

from ptlp.assignment import PassengerAssignment
from ptlp.instance import Instance, Line, Route


def example_instance() -> Instance:
    # Two services between the same stops, with different travel times.
    return Instance(
        name="single-assignment",
        nodes=pd.DataFrame({"id": [0, 1]}),
        od=pd.DataFrame({"k": [7], "origin": [0], "dest": [1], "demand": [100.0]}),
        lines=[Line(10, (0, 1), (10.0,), 0, 1, 20, 1),
               Line(20, (0, 1), (15.0,), 0, 1, 20, 1)],
        routes=[Route(100, 7, ((10, 0),), 10.0),
                Route(200, 7, ((20, 0),), 15.0)],
        horizon=1.0,
    )


def main() -> None:
    solver = PassengerAssignment(example_instance(), unmet_penalty=100.0)
    result = solver.solve(x=np.array([1, 1]), frequencies=np.array([3.0, 1.0]))
    if not result.feasible:
        raise RuntimeError(result.status)
    print("Route flows:", result.flows)
    print("Unserved by OD:", result.unserved)
    print("Travel time:", result.c_pass)
    print("Unserved penalty:", result.penalty_cost)
    print("Total objective:", result.objective)


if __name__ == "__main__":
    main()
