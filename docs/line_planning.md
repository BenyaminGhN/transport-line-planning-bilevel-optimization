# Line planning and solution search

This document describes the implemented line-planning layer that sits above
the passenger-assignment LP. It covers candidate-line construction, route
generation, plan evaluation, exact enumeration, and the genetic algorithm.

The current implementation treats each candidate line's frequency as fixed.
The leader decision is therefore the binary vector `x`, where `x[l] = 1`
enables a line and `x[l] = 0` disables it. Frequency optimization is not yet
part of the genetic representation.

## 1. Implemented workflow

The implemented data flow is:

```text
node and edge tables
    -> directed network graph
    -> candidate stop sequences
    -> candidate Line objects
    -> OD Route objects, including transfers
    -> Instance.build() sparse matrices
    -> PassengerAssignment.solve(x, frequencies)
    -> Evaluator.evaluate(x)
    -> exact enumeration or genetic algorithm
```

The passenger-assignment details and unmet-demand model are documented in
[passenger_assignment.md](passenger_assignment.md).

## 2. Candidate lines

Candidate-line functions are in `src/ptlp/lines.py`.

### Network input

`build_graph(nodes, edges)` expects:

- `nodes`: a DataFrame with an `id` column. Optional columns such as `x` and
  `y` are retained as node attributes.
- `edges`: a DataFrame with `u`, `v`, and `time` columns.

Edges are directed. Travel times must be finite and nonnegative. Duplicate
directed edges and edges that reference unknown nodes are rejected.

### Candidate generation

`generate_candidate_lines(graph, od, n_lines, min_stops, max_stops)` ranks
loop-free shortest paths between OD pairs. Higher-demand OD pairs are ranked
first, followed by lower travel time. The result is deterministic.

The function may return fewer than `n_lines` when the graph does not contain
enough distinct paths that satisfy the stop limits.

### Line parameters and CSV storage

`attach_line_parameters(stop_sequences, graph, params)` converts stop
sequences into `Line` objects. The parameter table requires:

- `cost`
- `frequency`
- `vehicle_capacity`
- `fleet`

An optional `id` column supplies line IDs; otherwise zero-based IDs are used.
`save_lines()` and `load_lines()` provide a stable CSV round trip for the
resulting candidates.

## 3. Passenger routes

Route functions are in `src/ptlp/routes.py`.

`build_line_layered_graph(lines)` creates one graph node per `(stop, line)`
pair. Riding arcs carry in-vehicle travel time, while transfer arcs connect
different lines at the same stop with zero cost. Waiting and transfer penalties
are therefore excluded from the current model.

`generate_routes(instance, k_routes)` returns up to `k_routes` distinct,
loop-free routes for each OD pair. A route can use multiple lines. OD pairs
that cannot be served receive no candidate route.

`attach_routes(instance, routes)` stores the routes and builds the sparse
capacity, OD, and route-line incidence matrices.

## 4. Plan evaluation

Plan evaluation is implemented in `src/ptlp/evaluate.py`.

For a binary line plan `x`, the evaluator performs the following steps:

1. Check the fixed line costs against `instance.budget`.
2. Check the fixed fleet requirements against `instance.fleet_total`.
3. Pass `x` and each line's fixed `Line.frequency` to the passenger-assignment
   solver.
4. Calculate passenger cost, including the unmet-demand penalty.
5. Calculate weighted travel-time Gini and P95 for served passengers.
6. Calculate the normalized authority objective.

The normalized objective is:

```text
lambda_op   * operating_cost / operating_cost_reference
+ lambda_pass * passenger_cost / passenger_cost_reference
```

`passenger_cost` includes both served-passenger in-vehicle time and the
unserved-demand penalty. Gini and P95 only use served passengers because an
unserved passenger has no observed route travel time.

Repeated evaluations of the same `x` are cached by default. The
`n_lp_solves` property counts actual passenger-assignment solves rather than
cache hits.

## 5. Exact enumeration

`enumerate_all(evaluator)` in `src/ptlp/exact.py` evaluates all `2^m` binary
plans for `m` candidate lines. It returns:

- the best feasible `EvaluationResult`;
- a tidy table containing every plan and its metrics;
- total runtime.

The default limit is 20 candidate lines. Enumeration is primarily intended as
a benchmark for small instances and for validating the genetic algorithm.

## 6. Genetic algorithm

The genetic algorithm is implemented in `src/ptlp/ga.py`. Each chromosome is
the binary line-selection vector `x`.

Implemented behavior includes:

- seeded, reproducible initialization;
- budget/fleet repair;
- tournament selection;
- one-point crossover;
- per-bit mutation;
- elitism;
- optional patience-based early stopping;
- one `GenerationRecord` callback per completed generation;
- population diversity and cumulative LP-solve tracking.

The initial population includes both the all-off and repaired all-on plans.
For small test instances, the GA result is compared directly with exact
enumeration.

## 7. Running the tests

Run commands from the repository root. The local environment created during
development is `.venv-codex`.

```powershell
.\.venv-codex\Scripts\Activate.ps1
```

Test candidate lines and routes:

```powershell
python -m pytest tests\test_lines_routes.py -v --basetemp .pytest-tmp
```

Test evaluation and exact enumeration:

```powershell
python -m pytest tests\test_evaluator_exact.py -v --basetemp .pytest-tmp
```

Test the genetic algorithm:

```powershell
python -m pytest tests\test_ga_implemented.py -v --basetemp .pytest-tmp
```

Test every currently implemented component:

```powershell
python -m pytest tests\test_assignment_penalty.py tests\test_lines_routes.py tests\test_evaluator_exact.py tests\test_ga_implemented.py -v --basetemp .pytest-tmp
```

Run the end-to-end example:

```powershell
python -m scripts.run_line_planning
```

The example builds a small network in memory, generates two candidate lines and
their OD routes, evaluates all four binary plans, runs the genetic algorithm,
and prints the GA gap to the exact optimum. It does not require files under
`data/`.

The current implemented suite reports `31 passed`. Running the entire project
suite additionally reports 17 skipped placeholder tests from the original
scaffold.

## 8. Current limitations and next steps

The following work remains:

- define and implement the actual synthetic and Mandl input formats in
  `src/ptlp/io.py`;
- add the real datasets under `data/synthetic` and `data/mandl`;
- decide whether frequency remains fixed or becomes a leader decision;
- if frequency becomes a decision, extend the GA chromosome, cache key,
  operating-cost model, and fleet constraints;
- implement network and algorithm visualizations;
- complete the synthetic, Mandl, and sensitivity experiment scripts;
- replace or remove the original skipped tests after reconciling them with the
  unmet-demand formulation.

Until real data formats are agreed, the automated tests use small hand-built
networks so that every expected result can be checked exactly.
