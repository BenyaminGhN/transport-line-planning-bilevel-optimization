"""Secondary distributional metrics. Never enter the objective."""
from __future__ import annotations

import numpy as np

from .instance import Instance


def passenger_travel_times(instance: Instance, flows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Input: y*. Output: (travel_time_i, weight_i) = (C_kr, y_kr) for routes with y>0."""
    raise NotImplementedError


def weighted_gini(tt: np.ndarray, w: np.ndarray) -> float:
    """Input: travel times and passenger weights. Output: G_w in [0, 1]."""
    raise NotImplementedError


def weighted_percentile(tt: np.ndarray, w: np.ndarray, q: float = 95.0) -> float:
    """Output: demand-weighted q-th percentile of travel time."""
    raise NotImplementedError
