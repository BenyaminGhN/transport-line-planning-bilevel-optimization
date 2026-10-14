# Line planning (the leader)

> 🔶 **CHANGED**: rewritten shorter and in plain words.

**Question answered:** which lines should the authority switch on?

The authority picks a 0/1 vector `x` (one entry per candidate line) that

* costs at most the **budget** `B`, and uses at most the **fleet** `F`,
* minimises `lambda_op x C_op/C_op_ref  +  lambda_pass x C_pass/C_pass_ref`,

where `C_pass` is the follower's objective (ride time + omega x unserved) **after** the passengers react to `x`.
That "choose, then passengers react" order is the Stackelberg game.

## Pipeline

    data files --io--> Instance --routes--> Instance.build()
    x --Evaluator--> check budget/fleet --> PassengerAssignment LP --> objective, Gini, P95, served share
    enumerate_all (exact)   and   GeneticAlgorithm      both call the same Evaluator

| File | Job |
|---|---|
| `instance.py` | data classes `Line`, `Route`, `Instance` (+ sparse matrices from `build()`) |
| `lines.py` | candidate lines from shortest paths between busy OD pairs |
| `routes.py` | up to `k_routes` routes per OD pair (at most 2 transfers) |
| `assignment.py` | the follower LP |
| `evaluate.py` | `Evaluator.evaluate(x)`: feasibility, LP, objective; caches results |
| `exact.py` | tries all 2^m plans: the exact benchmark |
| `ga.py` | genetic algorithm |
| `baselines.py` | cost-only and uncapacitated baselines |
| `viz/` | network picture, convergence, objective space, population, live dashboard, GIF |

## Candidate lines and routes

* A line is a path of stops. It can be **two-way** (`bidirectional`): then each direction has its own capacity.
* Lines come from shortest paths between OD pairs, busiest pairs first, one path per pair per round, so every busy
  pair gets a line before any pair gets a second one. A few connecting lines are added until every busy stop is
  served and the network is in one piece.
* Routes are found once with all lines on; `x` only switches them off later. A route may change lines at most
  `max_transfers` times and may not be slower than `detour_factor` x the fastest route.
* From the routes of one OD pair we keep `k_routes`: the fastest, then routes that use few lines (so the pair is
  still connected when only a few lines run), then the rest.

## Normalisation

`C_op_ref` = cost of all lines. `C_pass_ref` = passenger cost with all lines on and **no capacity limit**, using the
same `omega` as the follower. These are fixed numbers, so the objective is comparable between plans. They are stored
on the `Evaluator` (not in the instance), so two evaluators with different `omega` never mix.

## The genetic algorithm

1. Start with random plans (plus "nothing" and "everything"); repair any plan over budget/fleet by dropping the
   most expensive lines.
2. Repeat: keep the best 2 (elitism), pick parents by tournament, cross over at one point, flip each bit with
   probability `mutation_rate` (default `1.5 / number of lines`), repair, evaluate.
3. After **every** generation `on_generation(GenerationRecord)` is called: this feeds the dashboard.

Plans that break budget or fleet get `fitness = worst possible feasible objective + 1 + violation`, so the GA is
pushed back into the feasible region. Results are cached, so a repeated plan costs no LP.

Same seed gives the same run. Use several seeds (`--runs`) and report mean and standard deviation of the gap to the
exact optimum.

## Exact benchmark

`enumerate_all` evaluates every plan (budget/fleet-infeasible plans need no LP). Of several plans with the same
objective the first in binary order is returned, so the benchmark is deterministic. Up to about 20 lines is
practical (synthetic: 3 s; Mandl with 15 lines: about 2 minutes).

## Baselines

| Baseline | What it does | Note |
|---|---|---|
| Cost-only | cheapest plan | In the soft model an empty plan costs nothing, so the literal answer is "switch everything off". `min_served=0.5` gives the cheapest plan that serves at least half of the demand. |
| Uncapacitated | picks the best plan assuming nobody is limited by capacity, then judges it with capacity | Shows what ignoring capacity costs. |
