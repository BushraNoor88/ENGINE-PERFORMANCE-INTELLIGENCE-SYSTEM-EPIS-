# Jet Engine Cycle Analysis with ML-Based Parameter Calibration

A single-spool turbojet thermodynamic cycle model (Brayton cycle), built
with an object-oriented gas-path architecture, paired with two different
approaches — classical optimization and a machine learning surrogate — for
solving the **engine parameter calibration problem**: given observed
performance (thrust, fuel consumption), infer the underlying component
efficiencies.

This mirrors a real aerospace engineering workflow called **gas path
analysis**, used in engine health monitoring to detect degradation
(compressor fouling, turbine erosion, etc.) from in-service performance data.

## Why this problem matters

Component efficiencies (compressor, turbine, combustor, nozzle) aren't
fixed constants — they degrade with wear, and they're never perfectly known
even for a new engine. Being able to back them out from observable
performance data (thrust, TSFC) — rather than requiring instrumented rigs —
is directly useful for maintenance scheduling and fleet health monitoring.

## Architecture

```
engine/
├── atmosphere.py   # ISA standard atmosphere (validated against published tables)
├── components.py   # GasState + EngineComponent ABC -> Inlet, Compressor,
│                    # Combustor, Turbine, Nozzle (each with real thermodynamics)
└── engine.py        # TurbojetEngine: composes components into a full cycle,
                      # FlightCondition, PerformanceResult

calibration/
├── base.py           # ParameterCalibrator ABC, ObservedDataset
├── optimizer.py       # OptimizationCalibrator (scipy L-BFGS-B least squares)
└── ml_surrogate.py    # SurrogateMLCalibrator (Random Forest, trained on
                        # physics-model-generated synthetic data)

data/
└── reference_engine.py   # A representative small-turbojet configuration
                           # (J85-class: ~15 kN SLS thrust, PR=7, TIT=1200K)

tests/
└── test_components.py    # 15 unit/integration tests
```

**Design principles demonstrated:**
- **Abstraction**: `EngineComponent` and `ParameterCalibrator` are abstract
  base classes — new component types or calibration strategies can be added
  without touching `TurbojetEngine` or the calibration pipeline.
- **Composition over inheritance**: `TurbojetEngine` is *built from*
  component objects rather than inheriting from them — swap in a
  `HighBypassFan` in place of `Compressor` and nothing else changes.
- **Polymorphism**: `OptimizationCalibrator` and `SurrogateMLCalibrator`
  are interchangeable through the same `calibrate()` interface.

## Key physics implemented

- ISA standard atmosphere (troposphere + lower stratosphere)
- Isentropic compression/expansion relations with component efficiencies
- Combustor energy balance solved for fuel-air ratio (given a target
  turbine inlet temperature)
- Spool power balance (turbine work = compressor work)
- Choked vs. fully-expanded nozzle flow
- **Corrected mass flow scaling**: physical mass flow varies with
  altitude/Mach under a constant-corrected-flow assumption — this is what
  makes thrust properly "lapse" with altitude (~3x drop from sea-level
  static to cruise, matching real turbojet behavior) instead of staying
  artificially constant.

## Results

### Performance maps
Thrust and TSFC vs. Mach number at three altitudes, plus altitude lapse
rate at fixed Mach — see `thrust_tsfc_performance_map.png` and
`thrust_altitude_lapse.png`.

### Calibration: classical optimization vs. ML surrogate

A "degraded" engine (compressor efficiency 0.85→0.80, turbine efficiency
0.90→0.86, simulating wear) is measured at several flight conditions with
1.5-2% synthetic sensor noise added. Both calibrators try to recover the
hidden efficiencies:

| Parameter        | True   | Optimizer | Opt Err% | ML Surrogate | ML Err% |
|------------------|--------|-----------|----------|---------------|---------|
| inlet_recovery   | 0.9800 | 0.9965    | 1.69%    | 0.9859        | 0.61%   |
| compressor_eff   | 0.8000 | 0.7038    | 12.02%   | 0.7512        | 6.10%   |
| combustor_eff    | 0.9900 | 1.0000    | 1.01%    | 0.9704        | 1.98%   |
| turbine_eff      | 0.8600 | 0.9600    | 11.63%   | 0.9151        | 6.40%   |
| nozzle_eff       | 0.9700 | 0.9250    | 4.64%    | 0.9291        | 4.22%   |

**Key finding — parameter identifiability**: compressor and turbine
efficiency recover with noticeably higher error (~6-12%) than the other
three parameters. This isn't noise — it reflects a real structural
property of the problem: because turbine work is constrained to always
equal compressor work (spool balance), an error in one can be partly
compensated by an opposite error in the other while still matching the
observed thrust/TSFC. Adding more flight conditions did *not* resolve
this on its own — genuinely decoupling them would need an additional,
independent measurement (e.g. shaft speed or turbine-exit temperature).
This mirrors a documented limitation in real gas-path-analysis literature.

**Speed trade-off**: the optimizer is fast for a single engine (~46 ms)
because this physics model is cheap to evaluate. The ML surrogate's value
proposition is upfront training cost (~5 s, one-time) in exchange for
near-instant (~17 ms) repeated inference — useful for fleet-wide, real-time
health monitoring rather than one-off calibration.

## Running it

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v                  # 15 tests
python3 run_calibration_experiment.py         # calibration comparison (CLI)
python3 generate_performance_maps.py          # generates the PNG plots
```

## Interactive dashboard (Flask + live frontend)

A full web dashboard is included, backed by a real Flask API that calls
the *actual* `TurbojetEngine`, `OptimizationCalibrator`, and
`SurrogateMLCalibrator` classes above -- no physics or ML is reimplemented
in JavaScript. Every chart, readout, and calibration result on the page
comes from a live HTTP request to this same codebase.

```bash
python3 app.py
```

Then open **http://127.0.0.1:5000** in a browser. The ML surrogate is
trained once at server startup (a few seconds); after that, all requests
are near-instant.

**What's on it:**
- A live-animated gas-path schematic showing real station pressures/
  temperatures, driven by `/api/cycle`
- A Design Explorer (pressure ratio / turbine inlet temp sliders) driving
  `/api/mach-sweep` for the thrust/TSFC-vs-Mach charts
- A Sensitivity Lab (`/api/sensitivity-sweep`) showing the cost of
  compressor wear
- A Calibration Lab where you set a hidden "true" degraded engine,
  generate noisy field data via `/api/calibration/generate`, and run
  either the real scipy optimizer or the real trained Random Forest
  surrogate via `/api/calibration/run` -- side by side, with timing

If the backend isn't running, the page shows a clear banner rather than
failing silently or falling back to fake numbers.

## Known simplifications

- Calorically perfect gas (constant cp/gamma on each side of the combustor)
- Single-spool configuration only (no bypass/fan, no multi-spool)
- No compressor/turbine maps (efficiency treated as constant vs. operating
  point, not read off a real component map)
- No bleed air, variable geometry, or afterburner

These are standard simplifications for a first-pass "cold air standard"
cycle model (see Mattingly, *Elements of Gas Turbine Propulsion*) and keep
the model's assumptions transparent and auditable rather than hiding
complexity behind a black box.
