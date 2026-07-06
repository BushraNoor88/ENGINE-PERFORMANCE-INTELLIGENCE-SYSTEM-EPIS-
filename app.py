"""
Flask backend for the Engine Performance Intelligence dashboard.

This is a THIN API layer over the real, existing physics/ML code in
engine/, calibration/, and data/ -- it does not reimplement any physics.
Every number returned by these endpoints comes from actually calling
TurbojetEngine.run_cycle(), OptimizationCalibrator.calibrate(), or
SurrogateMLCalibrator.calibrate().

Run with:  python3 app.py
Then open: http://127.0.0.1:5000
"""

import time
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from engine.engine import TurbojetEngine, FlightCondition
from data.reference_engine import (
    build_reference_engine, TRUE_EFFICIENCIES, PRESSURE_RATIO,
    TURBINE_INLET_TEMP, AIR_MASS_FLOW, FUEL_HEATING_VALUE,
)
from calibration.base import ObservedDataset, ObservedDataPoint, PARAM_KEYS
from calibration.optimizer import OptimizationCalibrator
from calibration.ml_surrogate import SurrogateMLCalibrator

app = Flask(__name__, static_folder="webapp", static_url_path="")

# ---------------------------------------------------------------------
# ML surrogate is trained ONCE at server startup (this is the real,
# same-as-Python "train once, infer instantly" story from the CLI
# experiment) rather than retrained on every request.
# ---------------------------------------------------------------------
REFERENCE_CONDITIONS = [
    FlightCondition(altitude=0, mach=0.0),
    FlightCondition(altitude=0, mach=0.3),
    FlightCondition(altitude=5000, mach=0.6),
    FlightCondition(altitude=11000, mach=0.8),
]

print("Training ML surrogate calibrator at startup (one-time cost)...")
_t0 = time.perf_counter()
ml_calibrator = SurrogateMLCalibrator(reference_conditions=REFERENCE_CONDITIONS)
_ml_template_engine = build_reference_engine()
_n_used = ml_calibrator.train_on_simulated_data(_ml_template_engine, n_samples=3000)
print(f"  done in {time.perf_counter() - _t0:.2f}s ({_n_used} simulated samples)")


def _gas_state_dict(state) -> dict:
    return {
        "pressure_kpa": round(state.total_pressure / 1000, 2),
        "temperature_k": round(state.total_temperature, 1),
        "mass_flow_kg_s": round(state.mass_flow, 3),
    }


def _extract_efficiencies(payload: dict) -> dict:
    """Pulls efficiency overrides out of a request payload, falling back
    to the healthy reference values for anything not provided."""
    eff = dict(TRUE_EFFICIENCIES)
    for key in PARAM_KEYS:
        if key in payload:
            eff[key] = float(payload[key])
    return eff


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "ml_surrogate_trained": ml_calibrator.is_trained})


@app.route("/api/cycle", methods=["POST"])
def cycle():
    """
    Runs ONE real TurbojetEngine.run_cycle() call and returns the full
    result, including per-station pressure/temperature for the live
    schematic.
    """
    payload = request.get_json(force=True) or {}
    altitude = float(payload.get("altitude", 0.0))
    mach = float(payload.get("mach", 0.0))
    pressure_ratio = float(payload.get("pressure_ratio", PRESSURE_RATIO))
    turbine_inlet_temp = float(payload.get("turbine_inlet_temp", TURBINE_INLET_TEMP))
    efficiencies = _extract_efficiencies(payload)

    engine = build_reference_engine(efficiencies, pressure_ratio, turbine_inlet_temp)
    result = engine.run_cycle(FlightCondition(altitude=altitude, mach=mach))

    return jsonify({
        "thrust_kn": round(result.thrust / 1000, 3),
        "tsfc_kg_kgf_hr": round(result.tsfc * 3600 * 9.81, 4),
        "fuel_flow_kg_s": round(result.fuel_flow, 4),
        "is_choked": result.is_choked,
        "turbine_inlet_temp_k": round(result.turbine_inlet_temp, 1),
        "stations": {name: _gas_state_dict(s) for name, s in result.stations.items()},
    })


@app.route("/api/mach-sweep", methods=["POST"])
def mach_sweep():
    """
    Runs the real engine across a Mach sweep at 3 altitudes -- this
    directly powers the Design Explorer's thrust/TSFC-vs-Mach charts.
    """
    payload = request.get_json(force=True) or {}
    pressure_ratio = float(payload.get("pressure_ratio", PRESSURE_RATIO))
    turbine_inlet_temp = float(payload.get("turbine_inlet_temp", TURBINE_INLET_TEMP))
    efficiencies = _extract_efficiencies(payload)

    engine = build_reference_engine(efficiencies, pressure_ratio, turbine_inlet_temp)
    mach_values = [round(m, 2) for m in np.arange(0.05, 0.91, 0.05)]
    altitudes = [0, 5000, 11000]

    series = {}
    for alt in altitudes:
        thrusts, tsfcs = [], []
        for m in mach_values:
            result = engine.run_cycle(FlightCondition(altitude=alt, mach=m))
            thrusts.append(round(result.thrust / 1000, 3))
            tsfcs.append(round(result.tsfc * 3600 * 9.81, 4))
        series[str(alt)] = {"thrust_kn": thrusts, "tsfc": tsfcs}

    return jsonify({"mach": mach_values, "series": series})


@app.route("/api/sensitivity-sweep", methods=["POST"])
def sensitivity_sweep():
    """
    Sweeps compressor efficiency at sea-level-static and returns real
    thrust/TSFC at each point -- powers the Sensitivity Lab.
    """
    payload = request.get_json(force=True) or {}
    pressure_ratio = float(payload.get("pressure_ratio", PRESSURE_RATIO))
    turbine_inlet_temp = float(payload.get("turbine_inlet_temp", TURBINE_INLET_TEMP))

    eff_values = [round(e, 1) for e in np.arange(70, 92.1, 1.0)]
    thrusts, tsfcs = [], []
    for eff in eff_values:
        efficiencies = dict(TRUE_EFFICIENCIES)
        efficiencies["compressor_eff"] = eff / 100
        engine = build_reference_engine(efficiencies, pressure_ratio, turbine_inlet_temp)
        result = engine.run_cycle(FlightCondition(altitude=0, mach=0.0))
        thrusts.append(round(result.thrust / 1000, 3))
        tsfcs.append(round(result.tsfc * 3600 * 9.81, 4))

    return jsonify({"compressor_eff_pct": eff_values, "thrust_kn": thrusts, "tsfc": tsfcs})


@app.route("/api/calibration/generate", methods=["POST"])
def calibration_generate():
    """
    Builds a "true" degraded TurbojetEngine, runs it at several flight
    conditions, and adds synthetic sensor noise -- exactly mirroring
    run_calibration_experiment.py.
    """
    payload = request.get_json(force=True) or {}
    compressor_eff = float(payload.get("compressor_eff", 80.0)) / 100
    turbine_eff = float(payload.get("turbine_eff", 86.0)) / 100

    true_efficiencies = dict(TRUE_EFFICIENCIES)
    true_efficiencies["compressor_eff"] = compressor_eff
    true_efficiencies["turbine_eff"] = turbine_eff

    true_engine = build_reference_engine(true_efficiencies)

    rng = np.random.default_rng()  # fresh randomness each time -- real noise, not scripted
    observed = []
    for cond in REFERENCE_CONDITIONS:
        perf = true_engine.run_cycle(cond)
        noisy_thrust = perf.thrust * (1 + rng.normal(0, 0.015))
        noisy_tsfc = perf.tsfc * (1 + rng.normal(0, 0.02))
        observed.append({
            "altitude": cond.altitude, "mach": cond.mach,
            "thrust": noisy_thrust, "tsfc": noisy_tsfc,
        })

    return jsonify({"observed": observed, "true_efficiencies": true_efficiencies})


@app.route("/api/calibration/run", methods=["POST"])
def calibration_run():
    """
    Runs the REAL calibrator (scipy optimizer or trained ML surrogate)
    against observed data and returns recovered efficiencies + timing.
    """
    payload = request.get_json(force=True) or {}
    method = payload.get("method", "optimizer")
    observed_points = payload.get("observed", [])

    dataset = ObservedDataset()
    for p in observed_points:
        cond = FlightCondition(altitude=p["altitude"], mach=p["mach"])
        dataset.points.append(ObservedDataPoint(cond, p["thrust"], p["tsfc"]))

    t0 = time.perf_counter()
    if method == "ml":
        # Reuses the calibrator trained ONCE at server startup.
        recovered = ml_calibrator.calibrate(_ml_template_engine, dataset)
    else:
        engine = build_reference_engine()  # start from healthy-engine guess
        calibrator = OptimizationCalibrator()
        recovered = calibrator.calibrate(engine, dataset)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return jsonify({"recovered": recovered, "elapsed_ms": round(elapsed_ms, 2), "method": method})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
