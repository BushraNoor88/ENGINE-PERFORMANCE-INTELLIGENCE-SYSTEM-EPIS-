"""
Reference engine: a generic single-spool turbojet with parameters in the
same range as a small military turbojet (e.g. General Electric J85-class:
~13 kN sea-level static thrust, pressure ratio ~7:1, TIT ~1200 K).

These parameters are representative/textbook-typical values used for
validating the cycle model's output order of magnitude — NOT reverse
engineered from proprietary manufacturer data.
"""

from engine.components import Inlet, Compressor, Combustor, Turbine, Nozzle
from engine.engine import TurbojetEngine

# "True" efficiencies for a healthy, new engine — used to generate
# synthetic ground-truth data for the calibration experiments later.
TRUE_EFFICIENCIES = {
    "inlet_recovery": 0.98,
    "compressor_eff": 0.85,
    "combustor_eff": 0.99,
    "turbine_eff": 0.90,
    "nozzle_eff": 0.97,
}

FUEL_HEATING_VALUE = 43.0e6   # J/kg, Jet-A / Jet-A1 lower heating value
TURBINE_INLET_TEMP = 1200.0   # K, design-point TIT
PRESSURE_RATIO = 7.0          # overall compressor pressure ratio
AIR_MASS_FLOW = 20.0          # kg/s, design mass flow


def build_reference_engine(efficiencies: dict = None, pressure_ratio: float = None,
                            turbine_inlet_temp: float = None) -> TurbojetEngine:
    """
    Builds a TurbojetEngine instance. Pass a custom `efficiencies` dict
    (same keys as TRUE_EFFICIENCIES) to build a "degraded" or "hypothesis"
    engine for calibration experiments; defaults to the true/healthy engine.
    `pressure_ratio` and `turbine_inlet_temp` let callers explore different
    design points (used by the live dashboard's Design Explorer).
    """
    eff = efficiencies or TRUE_EFFICIENCIES
    pr = pressure_ratio if pressure_ratio is not None else PRESSURE_RATIO
    tit = turbine_inlet_temp if turbine_inlet_temp is not None else TURBINE_INLET_TEMP

    inlet = Inlet(efficiency=eff["inlet_recovery"], name="Inlet")
    compressor = Compressor(efficiency=eff["compressor_eff"],
                             pressure_ratio=pr, name="Compressor")
    combustor = Combustor(efficiency=eff["combustor_eff"],
                           fuel_heating_value=FUEL_HEATING_VALUE,
                           exit_temperature=tit, name="Combustor")
    turbine = Turbine(efficiency=eff["turbine_eff"], name="Turbine")
    nozzle = Nozzle(efficiency=eff["nozzle_eff"], name="Nozzle")

    return TurbojetEngine(inlet, compressor, combustor, turbine, nozzle,
                           air_mass_flow=AIR_MASS_FLOW)
