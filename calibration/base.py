"""
ParameterCalibrator: abstract interface for solving the "inverse problem" —
given observed engine performance (thrust, TSFC) at several flight
conditions, recover the underlying component efficiencies.

Two concrete strategies are provided:
  - OptimizationCalibrator: classical nonlinear least-squares (scipy)
  - SurrogateMLCalibrator:  train a regressor to map performance -> efficiencies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple

from engine.engine import TurbojetEngine, FlightCondition

# Order matters — used consistently to convert dict <-> vector everywhere.
PARAM_KEYS = ["inlet_recovery", "compressor_eff", "combustor_eff", "turbine_eff", "nozzle_eff"]

# Reasonable physical bounds for each efficiency (used by both calibrators)
PARAM_BOUNDS = {
    "inlet_recovery": (0.90, 1.00),
    "compressor_eff": (0.70, 0.95),
    "combustor_eff":  (0.90, 1.00),
    "turbine_eff":    (0.75, 0.96),
    "nozzle_eff":     (0.85, 1.00),
}


@dataclass
class ObservedDataPoint:
    """One (flight condition -> observed performance) measurement."""
    condition: FlightCondition
    observed_thrust: float   # N
    observed_tsfc: float     # kg/(N*s)


@dataclass
class ObservedDataset:
    """A collection of observed data points used to drive calibration."""
    points: List[ObservedDataPoint] = field(default_factory=list)

    def __iter__(self):
        return iter(self.points)

    def __len__(self):
        return len(self.points)

    def as_feature_matrix(self) -> List[List[float]]:
        """Flattens all points into a single feature vector (thrust, tsfc
        pairs concatenated across conditions) — used by the ML surrogate,
        which expects a FIXED set of flight conditions across all samples."""
        row = []
        for p in self.points:
            row.append(p.observed_thrust)
            row.append(p.observed_tsfc)
        return row


def params_to_vector(params: dict) -> List[float]:
    return [params[k] for k in PARAM_KEYS]


def vector_to_params(vector) -> dict:
    return {k: float(v) for k, v in zip(PARAM_KEYS, vector)}


class ParameterCalibrator(ABC):
    """Base class — different calibration strategies implement `calibrate`."""

    @abstractmethod
    def calibrate(self, engine: TurbojetEngine, observed_data: ObservedDataset) -> dict:
        """Returns a dict of best-fit efficiency parameters (see PARAM_KEYS)."""
        raise NotImplementedError
