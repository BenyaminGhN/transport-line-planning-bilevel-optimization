"""Small hand-made instances used by several tests.   >>> NEW FILE <<<"""
import pandas as pd

from ptlp.instance import Instance, Line, Route


def two_line_instance(cap_a=60.0, cap_b=100.0, time_a=10.0, time_b=15.0, demand=100.0, cost_a=6.0, cost_b=2.0,
                      budget=100.0, fleet_total=10, order=("A", "B")) -> Instance:
    """One OD pair 0 -> 1 served by two direct lines A and B (frequency 1/h, horizon 1 h, so capacity = cap).
    `order` is the order in which the two routes are listed (matters only for ties)."""
    lines = [Line(1, (0, 1), (time_a,), cost_a, 1.0, cap_a, 1), Line(2, (0, 1), (time_b,), cost_b, 1.0, cap_b, 1)]
    route = {"A": Route(0, 0, ((1, 0),), time_a), "B": Route(1, 0, ((2, 0),), time_b)}
    routes = [route[name] for name in order]
    routes = [Route(i, r.od_index, r.legs, r.cost) for i, r in enumerate(routes)]   # ids follow list order
    return Instance("two-line", pd.DataFrame({"id": [0, 1]}),
                    pd.DataFrame({"k": [0], "origin": [0], "dest": [1], "demand": [demand]}),
                    lines, routes, horizon=1.0, budget=budget, fleet_total=fleet_total)


def flows_by_line(instance, result):
    """{line id: passengers} for instances whose routes use one line each."""
    return {instance.routes[j].legs[0][0]: round(float(f), 6) for j, f in enumerate(result.flows)}
