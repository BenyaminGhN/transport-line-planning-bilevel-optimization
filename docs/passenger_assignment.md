# Passenger assignment (the follower)

> 🔶 **CHANGED**: rewritten shorter and in plain words; the maths is the same as before.

**Question answered:** the authority has switched some lines on. Where do the passengers go?

## The idea in one paragraph

Every OD pair (a start stop and an end stop) has a demand `D_k` and a few possible routes. Each passenger either
takes a route or is **left unserved**. Passengers choose so that the total of *ride time + penalty* is as small as
possible, but a route can only be used if all its lines are on, and a vehicle segment can only carry
`capacity x frequency x horizon` people. If the best route is full, the rest take the next best one; if nothing is
left, they are unserved and cost `omega` minutes each.

## The LP

    minimise   sum_k sum_r C_kr * y_kr   +   omega * sum_k u_k
    such that  sum_r y_kr + u_k = D_k                          every OD pair   (nobody disappears)
               y_kr <= D_k * x_l                               every line l of route r (line must be on)
               sum a_kr * y_kr <= kappa_l * f_l * H * x_l      every segment of line l  (capacity)
               y, u >= 0

| Symbol | Meaning | In the code |
|---|---|---|
| `y_kr` | passengers of OD pair k on route r | `AssignmentResult.flows` |
| `u_k` | unserved passengers of OD pair k | `AssignmentResult.unserved` |
| `C_kr` | ride time of the route (minutes) | `Route.cost` |
| `omega` | penalty per unserved passenger (minutes) | `unmet_penalty` |
| `kappa, f, H` | vehicle capacity, departures per hour, hours | `Line`, `Instance.horizon` |

The LP always has a solution (`u_k = D_k`, `y = 0` works), so the follower never "fails".

## What you get back

`PassengerAssignment(instance, unmet_penalty).solve(x, frequencies)` returns an `AssignmentResult`:

| Field | Meaning |
|---|---|
| `c_pass` | ride time of the served passengers only |
| `penalty_cost` | `omega x unserved` |
| `objective` | `c_pass + penalty_cost` (what the LP minimised) |
| `unserved_total`, `served_fraction` | `U(x)` and `1 - U(x) / total demand`  🔶 new |
| `segment_load`, `segment_util` | passengers per segment, and load / capacity |

## Choosing omega

`omega` is a number you set. By default it is **5 x the longest route** (`Instance.default_unmet_penalty`):
being left behind is then worse than any ride, but `omega` stays finite. If `omega` is smaller than a route's ride
time, passengers prefer not to travel at all. Check your choice with the omega sweep (`run_sensitivity`).

## Ties (several routes with the same cost)

> 🔶 **CHANGED**: new rule.

Each route gets a tiny extra cost `1e-6 x (its position in the route list of its OD pair)`. Among equal-cost routes
the one listed first is used, so the answer does not depend on solver luck. The reported `c_pass` uses the true
costs. Limit: this only orders routes *within one OD pair*; ties between different OD pairs that fight over the same
seats are left to the solver (which is deterministic, so the same input always gives the same output).

## Fairness numbers

`metrics.py` computes the Gini coefficient and the 95th percentile of ride time over **served passengers only**
(unserved people have no ride time). Always read them together with `served_fraction`.

## Run it

    python -m scripts.run_assignment
