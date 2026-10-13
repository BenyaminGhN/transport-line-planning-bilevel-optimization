"""Passenger-assignment LP with unmet demand, for fixed x and frequencies."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from .instance import Instance


@dataclass
class AssignmentResult:
    feasible: bool  # Optimum found; does NOT mean all demand was served.
    status: str
    flows: np.ndarray | None  # instance.routes order
    c_pass: float  # In-vehicle time only, without penalty
    segment_load: np.ndarray | None
    segment_util: np.ndarray | None  # NaN for zero capacity
    unserved: np.ndarray | None  # instance.od row order
    penalty_cost: float
    objective: float  # c_pass + penalty_cost
    fully_served: bool


class PassengerAssignment:
    """Reuse solve(x, frequencies) for each plan; do not mutate the instance.

    Both vectors follow instance.lines order, not line IDs. x=0 disables a line
    even with positive frequency. Frequencies are departures/hour and H is hours.
    Route time and unmet_penalty must use the same time unit (e.g. minutes).
    Any LP optimum may be returned; there is no tie-breaking perturbation.
    """

    def __init__(self, instance: Instance, unmet_penalty: float, capacitated: bool = True):
        if not np.isfinite(unmet_penalty) or unmet_penalty < 0:
            raise ValueError("unmet_penalty must be finite and nonnegative")
        instance.build()
        self.instance = instance
        self.unmet_penalty = float(unmet_penalty)
        self.capacitated = capacitated
        self._cost = np.concatenate((instance.route_cost, np.full(instance.n_od, unmet_penalty)))
        self._eq = sp.hstack((instance.A_od, sp.eye(instance.n_od)), format="csr")
        self._ub = sp.hstack((instance.A_cap,
                             sp.csr_matrix((len(instance.seg_line), instance.n_od))), format="csr")
        self._vehicle_capacity = np.asarray([line.vehicle_capacity for line in instance.lines])

    def solve(self, x: np.ndarray, frequencies: np.ndarray) -> AssignmentResult:
        """Budget/fleet checks belong to the caller. All-off means all unserved.

        Invalid input raises ValueError; solver failure is reported in status.
        Capacitated=False omits segment capacity, but still requires active service.
        """
        inst = self.instance
        x = np.asarray(x, dtype=float)
        frequencies = np.asarray(frequencies, dtype=float)
        if x.shape != (inst.n_lines,) or not np.all((x == 0) | (x == 1)):
            raise ValueError("x must be a binary vector with one entry per line")
        if frequencies.shape != x.shape or np.any(~np.isfinite(frequencies)) or np.any(frequencies < 0):
            raise ValueError("frequencies must be finite, nonnegative and match x")
        active = x * (frequencies > 0) # Lines that are actually running
        unavailable = np.asarray(inst.R_lines @ (1 - active)).ravel() > 0 # Routes that are unavailable due to inactive lines
        route_upper = np.where(unavailable, 0.0, inst.demand[inst.route_od]) # Upper bounds for route flows 0 <= ykr <= Dk if route is available, else 0
        capacity = (self._vehicle_capacity * frequencies * x * inst.horizon)[inst.seg_line] # Segment capacities for each segment, based on active lines and their frequencies e.q. kl*fl*xl*H
        if np.any(~np.isfinite(capacity)):
            raise ValueError("Computed capacity must be finite")
        upper = np.concatenate((route_upper, inst.demand)) # Upper bounds for the LP variables
        if len(self._cost) == 0:
            flows, unserved = np.zeros(inst.n_routes), np.zeros(inst.n_od)
            status = "Empty demand: optimal zero assignment"
        else:
            result = linprog(
                self._cost,
                A_ub=self._ub if self.capacitated and len(capacity) else None,
                b_ub=capacity if self.capacitated and len(capacity) else None,
                A_eq=self._eq if inst.n_od else None,
                b_eq=inst.demand if inst.n_od else None,
                bounds=np.column_stack((np.zeros_like(upper), upper)), method="highs",
            )
            if not result.success:
                return AssignmentResult(False, f"HiGHS {result.status}: {result.message}",
                                        None, np.inf, None, None, None, np.inf, np.inf, False)
            flows, unserved = result.x[:inst.n_routes], result.x[inst.n_routes:]
            status = result.message
        load = np.asarray(inst.A_cap @ flows).ravel()
        tol = 1e-7
        if (np.any(flows < -tol) or np.any(flows > route_upper + tol)
                or np.any(unserved < -tol) or np.any(unserved > inst.demand + tol)
                or not np.allclose(inst.A_od @ flows + unserved, inst.demand, rtol=0, atol=tol)
                or (self.capacitated and np.any(load > capacity + tol))):
            return AssignmentResult(False, "Solution failed residual validation", None,
                                    np.inf, None, None, None, np.inf, np.inf, False)
        utilization = np.divide(load, capacity, out=np.full_like(load, np.nan), where=capacity > 0)
        c_pass = float(inst.route_cost @ flows)
        penalty_cost = self.unmet_penalty * float(unserved.sum())
        return AssignmentResult(True, status, flows, c_pass, load, utilization,
                                unserved, penalty_cost, c_pass + penalty_cost,
                                bool(np.all(unserved <= tol)))
