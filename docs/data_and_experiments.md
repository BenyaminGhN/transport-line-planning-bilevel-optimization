# Data and experiments

> 🔶 **NEW FILE**

## Data

`python -m scripts.generate_data` writes the inputs. All random choices are seeded, so you get the same files every time.

| Folder | Files | Source |
|---|---|---|
| `data/synthetic/` | `nodes.csv`, `edges.csv`, `od_demand.csv`, `candidate_lines.csv`, `params.json` | generated |
| `data/netherlands/` | `source/graph.csv`, `source/demand.csv` (your generator's output), `nodes.csv`, `edges.csv`, `od_demand.csv`, `candidate_lines.csv`, `params.json` | your fake-data generator, converted by `datagen.import_network` |
| `data/mandl/` | `mandl1_nodes.txt`, `mandl1_links.txt`, `mandl1_demand.txt`, `candidate_lines.csv`, `params.json` | Mandl (1979) network as published in the [TransitNetworkDesign repository](https://github.com/RenatoArbex/TransitNetworkDesign) (Arbex & Cunha 2015); lines are generated |

**Synthetic city:** 3 x 4 grid, link times 2-12 minutes, 8 busy stops, 56 OD pairs (5759 passengers per hour), 10
candidate two-way lines (a few connecting lines may be added), 168 routes.

**Mandl:** 15 nodes, 21 links, 15570 trips per hour (checked against the published description). Lines are generated
with the same rules as the synthetic city (15 candidate lines, 234 routes).

**Line parameters (our assumptions, not given by Mandl):**
cost = 2 + 0.5 x one-way ride minutes; frequency 3 per hour; fleet = vehicles needed for the round trip;
**vehicle capacity** = the smallest capacity that lets "all lines on" carry all demand on its cheapest routes
(so smaller plans lose capacity); budget and fleet limit = 60 % of the totals; horizon H = 1 hour.
All of it is in `candidate_lines.csv` and `params.json`: edit the files to change it.

**Netherlands (your fake-data generator):** 20 Dutch cities, 29 links, all 380 OD pairs with 10-20 passengers per hour
(5805 in total). I checked it: the graph is connected, there are no self-pairs or duplicates, so it fits the problem.
What I adjusted:

* `distance` (100-2000, no unit) becomes **ride time in minutes = distance / 100** (1-20 min). Set `time_scale` in
  `import_network` to change it.
* Demand is read as **passengers per hour** (horizon H = 1 hour).
* The data has **no coordinates** and the links are random (Amsterdam-Dordrecht 4.8 min but Rotterdam-Arnhem 19.6 min),
  so the map is a computed layout, not geography. The city names stay in `nodes.csv`.
* The graph is sparse, so passengers need more transfers: `params.json` allows 4 transfers (detour factor 3). Even
  then 16 of 380 OD pairs have no route; their demand is always unserved, so the best possible served share is below 100 %.
* Because the graph is sparse (many cities hang on one link), 20 candidate lines are needed to reach every city, so
  the exhaustive benchmark would take about an hour (814 286 plans fit the budget). The script runs the GA only;
  use `--exact` if you want the benchmark.
* Your generator itself is not seeded (every run makes a new network). I kept its output as the fixed data set; if you
  re-run it, copy the new `graph.csv` and `demand.csv` into `data/netherlands/source/` and run
  `python -m scripts.generate_data`.

## Experiments

| Script | Shows | Output in `results/` |
|---|---|---|
| `run_synthetic` | GA vs exact optimum over several seeds, GIF of the search, baselines | `synthetic/` |
| `run_mandl` | the same on Mandl (GA only unless `--exact`) | `mandl/` |
| `run_netherlands` | GA on the 20-city network (about 40 s; `--exact` takes about an hour) | `netherlands/` |
| `run_baselines` | cost-only and uncapacitated vs Stackelberg for capacity x1, x0.7, x0.5 | `baselines_capacity.csv` |
| `run_sensitivity` | capacity, `lambda_op`, `omega`, budget, fleet, `k_routes`, capacity on/off | `sensitivity/*.csv, *.png` |

## Reading the dashboard

* **Top left, best network:** line width = passengers carried, colour = how full (white to dark red), dot size = demand.
* **Top right, convergence:** best so far, population mean, worst feasible plan; dashed green = exact optimum.
* **Bottom left, objective space:** every dot is a plan (x: operator cost, y: passenger cost, colour: served share).
  Grey = all plans (exact run). Red ring = GA best. Green star = exact optimum.
* **Bottom right, population:** one row per plan, one column per line; dark = on. Rows turn alike as the GA converges.

## What the first results say (synthetic city)

* The GA finds the exact optimum in every seed, using about 300 LP solves instead of 745.
* The optimum is limited by budget and fleet, not by the weights: changing `lambda_op` from 0.1 to 0.9, or `omega`
  from 1x to 20x the longest route, does not change the plan (only the objective value). Budget, fleet, capacity and
  the number of routes do change it.
* With generous capacity all methods give the same plan. The Stackelberg plan beats the capacity-blind plan once
  capacity is scarce (capacity x0.5: objective 2.28 vs 2.42).
