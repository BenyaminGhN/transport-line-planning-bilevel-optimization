"""Pure data containers. No optimisation logic lives here."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class Line:
    """Candidate transit line l in L.

    Segments are DIRECTED. A bidirectional line is stored as two Line objects
    (or one Line whose segment list contains both directions) - decide once in io.py.
    """
    id: int
    stops: tuple[int, ...]            # ordered node ids
    segment_times: tuple[float, ...]  # t0_ls, len = len(stops) - 1
    cost: float                       # c_l
    frequency: float                  # f_l  (vehicles per hour)
    vehicle_capacity: int             # kappa_l
    fleet: int                        # v_l

    @property
    def segments(self) -> tuple[tuple[int, int], ...]:
        """Output: ((u, v), ...) directed arcs of the line, length len(stops)-1."""
        return tuple(zip(self.stops[:-1], self.stops[1:]))

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
            values = np.asarray([*line.segment_times, line.frequency, line.vehicle_capacity], dtype=float)
            if np.any(~np.isfinite(values)) or np.any(values < 0):
                raise ValueError("Line times, frequency and vehicle capacity must be finite and nonnegative")
            for s in range(len(line.segments)):
                self.seg_index[line.id, s] = len(owners)
                owners.append(i)
                capacities.append(line.segment_capacity(self.horizon))
        self.seg_line = np.asarray(owners, dtype=int)
        self.seg_capacity = np.asarray(capacities, dtype=float)
        cap_rows, cap_cols, line_rows, line_cols, route_od = [], [], [], [], []
        for j, route in enumerate(self.routes):
            if route.od_index not in od_pos:
                raise ValueError("Route references an unknown OD ID")
            k = od_pos[route.od_index]
            if not route.legs or len(set(route.legs)) != len(route.legs):
                raise ValueError("Routes must be nonempty and cannot repeat a line segment")
            current = self.od.iloc[k]["origin"]
            travel_time = 0.0
            for line_id, s in route.legs:
                if (line_id, s) not in self.seg_index:
                    raise ValueError("Route references an unknown line segment")
                line = self.lines[line_pos[line_id]]
                start, end = line.segments[s]
                if start != current:
                    raise ValueError("Route legs must form a connected path from the OD origin")
                current = end
                travel_time += line.segment_times[s]
                cap_rows.append(self.seg_index[line_id, s])
                cap_cols.append(j)
            if current != self.od.iloc[k]["dest"]:
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
        raise NotImplementedError
