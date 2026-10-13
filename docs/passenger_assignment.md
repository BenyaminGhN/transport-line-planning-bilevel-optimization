# Isolated passenger assignment

This implemented step takes fixed binary line decisions `x`, fixed candidate
frequencies `f`, and previously prepared routes. It does not run the GA or check
the operator's budget/fleet. Line order is `instance.lines`; OD order is the row
order of `instance.od`; flow order is `instance.routes`. `Route.od_index` refers
to the OD table's `k` ID, which need not be a contiguous row number.

## Model

For each OD pair k, introduce unserved demand u_k >= 0:

    minimize sum(C_kr * y_kr) + w * sum(u_k)
    subject to
        sum_r(y_kr) + u_k = D_k                  for each OD
        y_kr <= D_k * x_l                       for each line used by a route
        sum(a_ls_kr * y_kr) <= kappa_l*f_l*H*x_l for each line segment
        y_kr >= 0, 0 <= u_k <= D_k

This is exactly equivalent to demand inequalities `sum_r(y_kr) <= D_k`
and adding `w * sum_k(D_k - sum_r(y_kr))` to the objective. Both x and f are
constants inside this solve, so the capacity expression does not make the LP
nonlinear. A route is disabled if any required line has x=0 or f=0.

H is in hours and f in departures/hour. Route costs and w are in minutes if
segment times are minutes; the total objective is in passenger-minutes.
Demand covers the same H. Capacity is aggregated over this horizon, not a
timetable-level guarantee. Transfers have no time penalty. Flow is continuous.

The penalty is a trade-off, not a strict promise to maximize passengers served.
If a route costs more than w, leaving its demand unserved may be optimal even
with spare capacity. Equal cost and penalty can give multiple optima. A strict
"maximize service, then minimize travel time" policy would require a two-stage
lexicographic objective; that is not the weighted model implemented here.

## Run

From the repository root, in an environment with the project dependencies:

    python -m pip install -e .
    python -m scripts.run_assignment

The example has demand 100, route times 10 and 15 minutes, vehicle capacity 20,
H=1, x=[1,1], f=[3,1], w=100. It returns flows [60,20], unserved [20], travel
cost 900, penalty 2000, total 2900.

Reuse the prepared solver for multiple independent candidate plans:

```python
from ptlp.assignment import PassengerAssignment
from scripts.run_assignment import example_instance

solver = PassengerAssignment(example_instance(), unmet_penalty=100.0)
result = solver.solve(x=[1, 1], frequencies=[3.0, 1.0])
if not result.feasible:
    raise RuntimeError(result.status)
print(result.flows, result.unserved, result.objective)
```

`feasible` means an optimal, numerically validated LP solution was found.
`fully_served` separately indicates zero unserved demand within tolerance.
Even all lines off is feasible under this model. `c_pass` excludes the penalty;
`objective` includes it. No deterministic tie-breaking is promised. Zero-capacity
segment utilization is NaN. Solver failures remain distinct from unmet demand.
The solver builds matrices once; do not mutate the instance afterwards.

## Files and integration boundary

- `src/ptlp/instance.py`: builds and validates route/OD/segment incidence matrices.
- `src/ptlp/assignment.py`: solves the LP and returns service, time and penalty.
- `scripts/run_assignment.py`: runnable, hand-made instance and one solve.
- `tests/test_assignment_penalty.py`: shortage, frequency, activation, shared
  capacity, transfer, missing routes, low penalty and validation tests.

`Line.frequency` remains as legacy/default instance data; the lower-level solve
uses its explicit `frequencies` argument. Static `seg_capacity` is not the
capacity of every candidate. Static cost/fleet fields are not recalculated here.

`Evaluator` and GA remain unimplemented. When integrating, pass both x and f,
cache on both vectors, and retain the penalty in the authority's evaluation
(with an explicitly chosen normalization). Do not reject a plan merely because
some demand is unserved. Budget/fleet relationships to frequency must be supplied
separately. Travel-time Gini/P95 for served passengers must be reported alongside
unserved demand; unserved passengers do not have an observed route travel time.
