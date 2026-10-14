"""Pure data containers. No optimisation logic lives here."""
from __future__ import annotations

from dataclasses import dataclass, field, replace  # >>> CHANGED: + replace (used by Instance.modified) <<<

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class Line:
    """Candidate transit line l in L.

    Segments are DIRECTED.
    >>> CHANGED: decided the open question of the old docstring. A line can run in one direction only
    (bidirectional=False, your original behaviour) or in both directions (bidirectional=True).
    A two-way line has 2*(n-1) segments: indexes 0..n-2 run forward (stops[i] -> stops[i+1]) and
    index (n-1)+i is the reverse of forward segment i. Each direction has its own capacity.
    Two-way lines are what an undirected network such as Mandl needs; otherwise every corridor would
    need two separate candidate lines and the search space would double. <<<
    """
    id: int
    stops: tuple[int, ...]            # ordered node ids
    segment_times: tuple[float, ...]  # t0_ls of the FORWARD segments, len = len(stops) - 1
    cost: float                       # c_l
    frequency: float                  # f_l  (vehicles per hour)
    vehicle_capacity: float           # kappa_l  # >>> CHANGED: int -> float (capacity sweeps give e.g. 28.5) <<<
    fleet: int                        # v_l
    bidirectional: bool = False       # >>> NEW <<<

    # >>> NEW: number of forward segments (helper for the two-way indexing above)
    @property
    def n_forward(self) -> int:
        return len(self.stops) - 1
    # <<< END NEW

    @property
    def segments(self) -> tuple[tuple[int, int], ...]:
        """Output: ((u, v), ...) directed arcs of the line (forward arcs, then reverse arcs if two-way)."""
        # >>> CHANGED: add the reverse arcs for two-way lines
        forward = tuple(zip(self.stops[:-1], self.stops[1:]))
        if not self.bidirectional:
            return forward
        return forward + tuple((v, u) for u, v in forward)
        # <<< END CHANGED

    # >>> NEW: in-vehicle time of segment s (a reverse segment takes as long as its forward twin)
    def segment_time(self, s: int) -> float:
        return self.segment_times[s % self.n_forward]
    # <<< END NEW

    def segment_capacity(self, horizon: float) -> float:
        """Input: H. Output: kappa_l * f_l * H (passengers per segment over the horizon)."""
        return self.vehicle_capacity * self.frequency * horizon


@dataclass(frozen=True)
class Route:
    """Passenger route r in R_k: ordered legs, possibly with transfers."""
    id: int
    od_index: int                      # k
    legs: tuple[tuple[int, int], ...]  # ((line_id, segment_idx), ...) in travel order
    cost: float                        # C_kr = sum of t0_ls (in-vehicle time only)

    @property
    def lines(self) -> frozenset[int]:
        """Output: L(r), the set of line ids used by this route."""
        return frozenset(line_id for line_id, _ in self.legs)


@dataclass
class Instance:
    """Everything the optimiser needs. Call build() once after routes are attached."""
    name: str
    nodes: pd.DataFrame                # columns: id, x, y   (x,y optional for Mandl -> layout fallback)
    od: pd.DataFrame                   # columns: k, origin, dest, demand (D_k)
    lines: list[Line]
    routes: list[Route] = field(default_factory=list)
    horizon: float = 1.0               # H
    budget: float = np.inf             # B
    fleet_total: int = 10**9           # F
    lambda_op: float = 0.5
    lambda_pass: float = 0.5
    # >>> NEW: physical links (u, v, time), only used for drawing the network and for line generation
    edges: pd.DataFrame | None = None
    # >>> NEW: omega of the PDF (penalty per unserved passenger, in minutes). None = default_unmet_penalty()
    unmet_penalty: float | None = None

    # ---- derived arrays, filled by build() --------------------------------
    seg_index: dict[tuple[int, int], int] = field(default_factory=dict)  # (line_id, seg_idx) -> row
    seg_line: np.ndarray | None = None       # (n_seg,)   line index owning each segment
    seg_capacity: np.ndarray | None = None   # (n_seg,)   kappa*f*H at x_l = 1
    A_cap: sp.csr_matrix | None = None       # (n_seg, n_routes)  a^{ls}_{kr}
    A_od: sp.csr_matrix | None = None        # (n_od,  n_routes)  route belongs to OD k
    R_lines: sp.csr_matrix | None = None     # (n_routes, n_lines) 1 if route uses line
    route_cost: np.ndarray | None = None     # (n_routes,) C_kr
    route_od: np.ndarray | None = None       # (n_routes,) k of each route
    demand: np.ndarray | None = None         # (n_od,) D_k
    line_cost: np.ndarray | None = None      # (n_lines,) c_l
    line_fleet: np.ndarray | None = None     # (n_lines,) v_l
    c_op_ref: float | None = None
    c_pass_ref: float | None = None

    @property
    def n_lines(self) -> int: return len(self.lines)
    @property
    def n_routes(self) -> int: return len(self.routes)
    @property
    def n_od(self) -> int: return len(self.od)

    def build(self) -> None:
        """Input: lines + routes already set. Output: None (fills all derived arrays above)."""
        if not np.isfinite(self.horizon) or self.horizon <= 0:
            raise ValueError("horizon must be finite and positive")
        line_ids = [line.id for line in self.lines]
        od_ids = list(self.od["k"])
        if len(set(line_ids)) != len(line_ids) or len(set(od_ids)) != len(od_ids):
            raise ValueError("Line and OD IDs must be unique")
        if len({route.id for route in self.routes}) != self.n_routes:
            raise ValueError("Route IDs must be unique")
        line_pos = {key: i for i, key in enumerate(line_ids)}
        od_pos = {key: i for i, key in enumerate(od_ids)}
        self.demand = self.od["demand"].to_numpy(dtype=float)
        if np.any(~np.isfinite(self.demand)) or np.any(self.demand < 0):
            raise ValueError("Demand must be finite and nonnegative")
        self.seg_index = {}
        owners, capacities = [], []
        for i, line in enumerate(self.lines):
            if len(line.stops) < 2 or len(line.segment_times) != len(line.stops) - 1:
                raise ValueError("Each line needs stops and matching segment times")
            values = np.asarray([
                *line.segment_times, line.cost, line.frequency, line.vehicle_capacity, line.fleet
            ], dtype=float)
            if np.any(~np.isfinite(values)) or np.any(values < 0):
                raise ValueError("Line times and parameters must be finite and nonnegative")
            for s in range(len(line.segments)):
                self.seg_index[line.id, s] = len(owners)
                owners.append(i)
                capacities.append(line.segment_capacity(self.horizon))
        self.seg_line = np.asarray(owners, dtype=int)
        self.seg_capacity = np.asarray(capacities, dtype=float)
        cap_rows, cap_cols, line_rows, line_cols, route_od = [], [], [], [], []
        # >>> CHANGED (speed only): look arrays up once instead of calling line.segments / od.iloc for
        # every leg of every route. The checks below are exactly yours. <<<
        origins, destinations = self.od["origin"].to_numpy(), self.od["dest"].to_numpy()
        arcs_of_line = {line.id: line.segments for line in self.lines}
        for j, route in enumerate(self.routes):
            if route.od_index not in od_pos:
                raise ValueError("Route references an unknown OD ID")
            k = od_pos[route.od_index]
            if not route.legs or len(set(route.legs)) != len(route.legs):
                raise ValueError("Routes must be nonempty and cannot repeat a line segment")
            current = origins[k]
            travel_time = 0.0
            for line_id, s in route.legs:
                if (line_id, s) not in self.seg_index:
                    raise ValueError("Route references an unknown line segment")
                line = self.lines[line_pos[line_id]]
                start, end = arcs_of_line[line_id][s]
                if start != current:
                    raise ValueError("Route legs must form a connected path from the OD origin")
                current = end
                travel_time += line.segment_time(s)  # >>> CHANGED: segment_time() also works for reverse segments <<<
                cap_rows.append(self.seg_index[line_id, s])
                cap_cols.append(j)
            if current != destinations[k]:
                raise ValueError("Route must end at the OD destination")
            if not np.isfinite(route.cost) or route.cost < 0 or not np.isclose(route.cost, travel_time):
                raise ValueError("Route cost must equal its total in-vehicle time")
            for line_id in sorted(route.lines):
                line_rows.append(j)
                line_cols.append(line_pos[line_id])
            route_od.append(k)
        self.route_od = np.asarray(route_od, dtype=int)
        self.route_cost = np.asarray([route.cost for route in self.routes], dtype=float)
        self.A_cap = sp.csr_matrix((np.ones(len(cap_rows)), (cap_rows, cap_cols)),
                                   shape=(len(owners), self.n_routes))
        self.A_od = sp.csr_matrix((np.ones(self.n_routes), (self.route_od, np.arange(self.n_routes))),
                                  shape=(self.n_od, self.n_routes))
        self.R_lines = sp.csr_matrix((np.ones(len(line_rows)), (line_rows, line_cols)),
                                     shape=(self.n_routes, self.n_lines))
        self.line_cost = np.asarray([line.cost for line in self.lines], dtype=float)
        self.line_fleet = np.asarray([line.fleet for line in self.lines], dtype=float)

    def set_references(self, c_op_ref: float, c_pass_ref: float) -> None:
        """Fix the normalisation constants (constant across all plans)."""
        values = np.asarray([c_op_ref, c_pass_ref], dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("reference values must be finite and positive")
        self.c_op_ref = float(c_op_ref)
        self.c_pass_ref = float(c_pass_ref)

    # >>> NEW: three small helpers (default omega, demand per node for plots, copy for sensitivity runs)
    def default_unmet_penalty(self, factor: float = 5.0) -> float:
        """Default omega: `factor` times the longest route (in minutes). Being left behind then costs more
        than any single ride, but omega stays finite as the PDF requires. Check it with the omega sweep."""
        longest = float(np.max(self.route_cost)) if self.routes else 1.0
        return factor * max(longest, 1.0)

    def node_demand(self) -> pd.Series:
        """Total demand (origin + destination) per node id, used for node sizes in the plots."""
        total = self.od.groupby("origin")["demand"].sum().add(
            self.od.groupby("dest")["demand"].sum(), fill_value=0.0)
        return total.reindex(self.nodes["id"]).fillna(0.0)

    def modified(self, capacity_scale: float = 1.0, budget: float | None = None,
                 fleet_total: float | None = None, lambda_op: float | None = None) -> "Instance":
        """Copy with other vehicle capacity / budget / fleet / weight (for sensitivity analysis).
        The original instance is not touched; routes are shared."""
        lines = [replace(line, vehicle_capacity=line.vehicle_capacity * capacity_scale) for line in self.lines]
        new = replace(self, lines=lines,
                      budget=self.budget if budget is None else budget,
                      fleet_total=self.fleet_total if fleet_total is None else fleet_total)
        if lambda_op is not None:
            new.lambda_op, new.lambda_pass = float(lambda_op), 1.0 - float(lambda_op)
        new.build()
        return new
    # <<< END NEW
