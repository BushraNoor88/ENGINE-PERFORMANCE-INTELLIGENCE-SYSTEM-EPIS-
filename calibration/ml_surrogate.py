"""
SurrogateMLCalibrator: trains a regressor to directly map an observed
performance signature (thrust/TSFC at a fixed set of flight conditions) to
component efficiencies -- the inverse of what the physics model computes.

Motivation: OptimizationCalibrator re-runs the full cycle model dozens of
times per calibration (slow if you need to calibrate many engines, e.g. a
whole fleet, or do it in near-real-time from onboard sensors). Once
trained, this surrogate calibrates a new engine in a single forward pass.
"""

import copy
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from calibration.base import (
    ParameterCalibrator, ObservedDataset, ObservedDataPoint,
    PARAM_KEYS, PARAM_BOUNDS, params_to_vector, vector_to_params,
)
from engine.engine import TurbojetEngine, FlightCondition


class SurrogateMLCalibrator(ParameterCalibrator):
    """
    Supervised approach to the inverse problem:
      1. Sample many random efficiency combinations.
      2. Run the (fast, deterministic) physics model to get their
         resulting (thrust, TSFC) at a FIXED set of flight conditions.
      3. Train a regressor: performance_vector -> efficiencies.
      4. At inference time, feed in observed performance and get an
         instant efficiency estimate (no iterative optimization).
    """

    def __init__(self, reference_conditions: list, n_estimators: int = 300,
                 random_state: int = 42):
        # A fixed set of flight conditions the model is always evaluated at
        # -- the ML model's input feature vector has fixed length/order, so
        # training and inference must use the exact same conditions.
        self.reference_conditions = reference_conditions
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )
        self.is_trained = False
        self._rng = np.random.default_rng(random_state)

    def _random_efficiency_sample(self) -> dict:
        return {k: self._rng.uniform(*PARAM_BOUNDS[k]) for k in PARAM_KEYS}

    def _simulate_performance_vector(self, engine: TurbojetEngine, params: dict) -> list:
        engine.set_calibratable_params(params)
        vector = []
        for cond in self.reference_conditions:
            result = engine.run_cycle(cond)
            vector.extend([result.thrust, result.tsfc])
        return vector

    def train_on_simulated_data(self, engine_template: TurbojetEngine, n_samples: int = 3000):
        """
        Generates synthetic (efficiencies -> performance) training pairs by
        randomly sampling efficiency combos and running the physics model.
        This is the "inverse problem via supervised learning" trick: the
        forward model (physics) generates labeled data for free, since it's
        cheap and deterministic -- no real measurements needed for training.
        """
        # Work on a copy so we don't leave the caller's engine mutated.
        engine = copy.deepcopy(engine_template)

        X, y = [], []
        for _ in range(n_samples):
            sampled_effs = self._random_efficiency_sample()
            try:
                perf_vector = self._simulate_performance_vector(engine, sampled_effs)
            except ValueError:
                # Some random efficiency combos can produce non-physical
                # combustor solutions (e.g. TIT unreachable) -- skip those.
                continue
            X.append(perf_vector)
            y.append(params_to_vector(sampled_effs))

        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        return len(X)

    def calibrate(self, engine: TurbojetEngine, observed_data: ObservedDataset) -> dict:
        if not self.is_trained:
            raise RuntimeError("Call train_on_simulated_data() before calibrate().")

        # observed_data must contain measurements at exactly
        # self.reference_conditions, in the same order, for the feature
        # vector to line up with what the model was trained on.
        feature_vector = np.array(observed_data.as_feature_matrix()).reshape(1, -1)
        predicted_vector = self.model.predict(feature_vector)[0]
        return vector_to_params(predicted_vector)
