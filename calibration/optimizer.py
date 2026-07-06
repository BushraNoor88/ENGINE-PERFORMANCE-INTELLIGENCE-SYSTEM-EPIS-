"""
OptimizationCalibrator: classical nonlinear least-squares parameter
identification using scipy.optimize. Iteratively adjusts efficiencies so
the physics model's predicted thrust/TSFC matches observed data.
"""

import numpy as np
from scipy.optimize import minimize

from calibration.base import (
    ParameterCalibrator, ObservedDataset, PARAM_KEYS, PARAM_BOUNDS,
    params_to_vector, vector_to_params,
)
from engine.engine import TurbojetEngine


class OptimizationCalibrator(ParameterCalibrator):
    """
    Fits component efficiencies by minimizing squared error between
    predicted and observed (thrust, TSFC) across all provided flight
    conditions. Uses L-BFGS-B (supports simple bounds, no need for
    gradients since scipy will estimate them numerically).
    """

    def __init__(self, tsfc_weight: float = 1.0e6, max_iter: int = 200):
        # TSFC values are ~1e-4 in SI units (kg/(N*s)) while thrust is ~1e4 N,
        # so TSFC error terms need a large weight or they're invisible next
        # to thrust error terms in the combined loss.
        self.tsfc_weight = tsfc_weight
        self.max_iter = max_iter
        self.last_result = None  # scipy OptimizeResult, kept for diagnostics

    def calibrate(self, engine: TurbojetEngine, observed_data: ObservedDataset,
                  initial_guess: dict = None) -> dict:
        x0_params = initial_guess or {k: sum(PARAM_BOUNDS[k]) / 2 for k in PARAM_KEYS}
        x0 = np.array(params_to_vector(x0_params))
        bounds = [PARAM_BOUNDS[k] for k in PARAM_KEYS]

        def loss(vector):
            params = vector_to_params(vector)
            engine.set_calibratable_params(params)
            total_error = 0.0
            for point in observed_data:
                predicted = engine.run_cycle(point.condition)
                thrust_error = (predicted.thrust - point.observed_thrust) ** 2
                tsfc_error = (predicted.tsfc - point.observed_tsfc) ** 2 * self.tsfc_weight
                total_error += thrust_error + tsfc_error
            return total_error

        result = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": self.max_iter})
        self.last_result = result

        return vector_to_params(result.x)
