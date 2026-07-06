"""
End-to-end validation experiment:

1. Define a "true" degraded engine (efficiencies different from the healthy
   baseline -- simulating wear/aging).
2. Generate synthetic "observed" performance data at several flight
   conditions, with sensor noise added (realistic: you never get perfect
   measurements).
3. Hide the true efficiencies. Try to recover them using:
     a) OptimizationCalibrator (classical scipy least-squares)
     b) SurrogateMLCalibrator (random forest trained on simulated data)
4. Compare recovered efficiencies vs ground truth, and compare wall-clock
   time for each approach.
"""

import time
import copy
import numpy as np

from data.reference_engine import build_reference_engine, TRUE_EFFICIENCIES
from engine.engine import FlightCondition
from calibration.base import ObservedDataset, ObservedDataPoint, PARAM_KEYS
from calibration.optimizer import OptimizationCalibrator
from calibration.ml_surrogate import SurrogateMLCalibrator

rng = np.random.default_rng(7)

# ---------------------------------------------------------------
# 1. Define a "true" DEGRADED engine (simulating an aged engine with
#    lower compressor/turbine efficiency than the healthy baseline).
# ---------------------------------------------------------------
TRUE_DEGRADED_EFFICIENCIES = dict(TRUE_EFFICIENCIES)
TRUE_DEGRADED_EFFICIENCIES["compressor_eff"] = 0.80   # was 0.85 when new
TRUE_DEGRADED_EFFICIENCIES["turbine_eff"] = 0.86      # was 0.90 when new

true_engine = build_reference_engine(TRUE_DEGRADED_EFFICIENCIES)

# ---------------------------------------------------------------
# 2. Generate noisy synthetic observed data at a handful of flight
#    conditions (simulating flight test / in-service sensor readings).
# ---------------------------------------------------------------
REFERENCE_CONDITIONS = [
    FlightCondition(altitude=0, mach=0.0),
    FlightCondition(altitude=0, mach=0.3),
    FlightCondition(altitude=5000, mach=0.6),
    FlightCondition(altitude=11000, mach=0.8),
]

NOISE_STD_THRUST = 0.015   # 1.5% measurement noise on thrust
NOISE_STD_TSFC = 0.02      # 2% measurement noise on TSFC

observed = ObservedDataset()
for cond in REFERENCE_CONDITIONS:
    true_perf = true_engine.run_cycle(cond)
    noisy_thrust = true_perf.thrust * (1 + rng.normal(0, NOISE_STD_THRUST))
    noisy_tsfc = true_perf.tsfc * (1 + rng.normal(0, NOISE_STD_TSFC))
    observed.points.append(ObservedDataPoint(cond, noisy_thrust, noisy_tsfc))

print("=" * 70)
print("GROUND TRUTH (hidden from calibrators):")
for k in PARAM_KEYS:
    print(f"  {k:18s}: {TRUE_DEGRADED_EFFICIENCIES[k]:.4f}")
print("=" * 70)

# ---------------------------------------------------------------
# 3a. Classical optimization calibration
# ---------------------------------------------------------------
opt_engine = build_reference_engine()  # start from healthy-engine guess
opt_calibrator = OptimizationCalibrator()

t0 = time.perf_counter()
opt_result = opt_calibrator.calibrate(opt_engine, observed)
opt_time = time.perf_counter() - t0

# ---------------------------------------------------------------
# 3b. ML surrogate calibration
# ---------------------------------------------------------------
ml_calibrator = SurrogateMLCalibrator(reference_conditions=REFERENCE_CONDITIONS)
ml_template_engine = build_reference_engine()

t0 = time.perf_counter()
n_used = ml_calibrator.train_on_simulated_data(ml_template_engine, n_samples=3000)
train_time = time.perf_counter() - t0

t0 = time.perf_counter()
ml_result = ml_calibrator.calibrate(ml_template_engine, observed)
ml_inference_time = time.perf_counter() - t0

# ---------------------------------------------------------------
# 4. Compare results
# ---------------------------------------------------------------
print()
print(f"{'Parameter':18s} {'True':>8s} {'Optimizer':>10s} {'Opt Err%':>9s} {'ML':>8s} {'ML Err%':>9s}")
print("-" * 70)
for k in PARAM_KEYS:
    true_val = TRUE_DEGRADED_EFFICIENCIES[k]
    opt_val = opt_result[k]
    ml_val = ml_result[k]
    opt_err = abs(opt_val - true_val) / true_val * 100
    ml_err = abs(ml_val - true_val) / true_val * 100
    print(f"{k:18s} {true_val:8.4f} {opt_val:10.4f} {opt_err:8.2f}% {ml_val:8.4f} {ml_err:8.2f}%")

print()
print(f"Optimizer calibration time  : {opt_time*1000:8.1f} ms  (iterative, re-runs physics model each step)")
print(f"ML surrogate TRAINING time  : {train_time*1000:8.1f} ms  ({n_used} simulated samples, one-time cost)")
print(f"ML surrogate INFERENCE time : {ml_inference_time*1000:8.2f} ms  (instant -- reusable for future engines)")
