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
        raise NotImplementedError

    def segment_capacity(self, horizon: float) -> float:
        """Input: H. Output: kappa_l * f_l * H (passengers per segment over the horizon)."""
        raise NotImplementedError


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
        raise NotImplementedError


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
    def n_lines(self) -> int: raise NotImplementedError
    @property
    def n_routes(self) -> int: raise NotImplementedError
    @property
    def n_od(self) -> int: raise NotImplementedError

    def build(self) -> None:
        """Input: lines + routes already set. Output: None (fills all derived arrays above)."""
        raise NotImplementedError

    def set_references(self, c_op_ref: float, c_pass_ref: float) -> None:
        """Fix the normalisation constants (constant across all plans)."""
        raise NotImplementedError
