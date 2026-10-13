"""Secondary distributional metrics. Never enter the objective."""
from __future__ import annotations

import numpy as np

from .instance import Instance


def passenger_travel_times(instance: Instance, flows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Input: y*. Output: (travel_time_i, weight_i) = (C_kr, y_kr) for routes with y>0."""
    flows = np.asarray(flows, dtype=float)
    if flows.shape != (instance.n_routes,):
        raise ValueError("flows must have one entry per route")
    if np.any(~np.isfinite(flows)) or np.any(flows < -1e-9):
        raise ValueError("flows must be finite and nonnegative")
    if instance.route_cost is None:
        raise ValueError("instance.build() must be called before computing metrics")
    used = flows > 1e-9
    return instance.route_cost[used].copy(), flows[used].copy()


def weighted_gini(tt: np.ndarray, w: np.ndarray) -> float:
    """Input: travel times and passenger weights. Output: G_w in [0, 1]."""
    tt, w = _validated_samples(tt, w)
    positive = w > 0
    tt, w = tt[positive], w[positive]
    if not len(tt):
        return float("nan")
    order = np.argsort(tt, kind="stable")
    tt, w = tt[order], w[order]
    total_weight = float(w.sum())
    weighted_sum = float(tt @ w)
    if weighted_sum == 0:
        return 0.0
    previous_weight = np.cumsum(w) - w
    previous_weighted_time = np.cumsum(tt * w) - tt * w
    pairwise_half = float(np.sum(w * (tt * previous_weight - previous_weighted_time)))
    return float(np.clip(pairwise_half / (total_weight * weighted_sum), 0.0, 1.0))


def weighted_percentile(tt: np.ndarray, w: np.ndarray, q: float = 95.0) -> float:
    """Output: demand-weighted q-th percentile of travel time."""
    tt, w = _validated_samples(tt, w)
    if not np.isfinite(q) or not 0 <= q <= 100:
        raise ValueError("q must lie in [0, 100]")
    positive = w > 0
    tt, w = tt[positive], w[positive]
    if not len(tt):
        return float("nan")
    order = np.argsort(tt, kind="stable")
    tt, w = tt[order], w[order]
    threshold = q / 100.0 * float(w.sum())
    index = int(np.searchsorted(np.cumsum(w), threshold, side="left"))
    return float(tt[min(index, len(tt) - 1)])


def _validated_samples(tt: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tt = np.asarray(tt, dtype=float)
    w = np.asarray(w, dtype=float)
    if tt.ndim != 1 or w.ndim != 1 or tt.shape != w.shape:
        raise ValueError("travel times and weights must be one-dimensional arrays of equal length")
    if np.any(~np.isfinite(tt)) or np.any(~np.isfinite(w)):
        raise ValueError("travel times and weights must be finite")
    if np.any(tt < 0) or np.any(w < 0):
        raise ValueError("travel times and weights must be nonnegative")
    return tt, w
