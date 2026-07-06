"""
TurbojetEngine: composes individual components into a full gas-path cycle
and computes overall performance (thrust, TSFC, fuel flow).
"""

import math
from dataclasses import dataclass

from engine.atmosphere import StandardAtmosphere
from engine.components import GasState, Inlet, Compressor, Combustor, Turbine, Nozzle, CP_AIR, GAMMA_AIR


@dataclass
class FlightCondition:
    """Defines the operating point: altitude + Mach number."""
    altitude: float   # m
    mach: float

    def freestream_state(self, mass_flow: float) -> GasState:
        """Computes freestream stagnation (total) conditions for a given Mach number."""
        atmo = StandardAtmosphere.at_altitude(self.altitude)
        gamma = GAMMA_AIR

        temp_ratio = 1 + (gamma - 1) / 2 * self.mach ** 2
        total_temperature = atmo.temperature * temp_ratio
        total_pressure = atmo.pressure * temp_ratio ** (gamma / (gamma - 1))

        return GasState(
            total_pressure=total_pressure,
            total_temperature=total_temperature,
            mass_flow=mass_flow,
            cp=CP_AIR,
            gamma=GAMMA_AIR,
        )

    def ambient_pressure(self) -> float:
        return StandardAtmosphere.at_altitude(self.altitude).pressure

    def flight_speed(self) -> float:
        return StandardAtmosphere.at_altitude(self.altitude).speed_of_sound * self.mach


@dataclass
class PerformanceResult:
    """Summary of overall engine performance at one operating point."""
    thrust: float           # N
    tsfc: float             # kg/(N*s)  -- thrust specific fuel consumption
    fuel_flow: float        # kg/s
    exit_velocity: float    # m/s
    is_choked: bool
    turbine_inlet_temp: float  # K, useful diagnostic (material limit check)
    stations: dict = None   # station name -> GasState, for diagnostics/visualization

    def __repr__(self):
        return (f"PerformanceResult(thrust={self.thrust/1000:.2f}kN, "
                f"TSFC={self.tsfc*3600*9.81:.3f}kg/(kgf*hr), "
                f"fuel_flow={self.fuel_flow:.3f}kg/s, "
                f"choked={self.is_choked})")


class TurbojetEngine:
    """
    Composes Inlet -> Compressor -> Combustor -> Turbine -> Nozzle into a
    single-spool turbojet cycle. Uses composition (not inheritance) so that
    swapping in different component implementations (e.g. a variable-geometry
    nozzle) requires no change to this class.
    """

    def __init__(self, inlet: Inlet, compressor: Compressor, combustor: Combustor,
                 turbine: Turbine, nozzle: Nozzle, air_mass_flow: float,
                 design_condition: FlightCondition = None):
        self.inlet = inlet
        self.compressor = compressor
        self.combustor = combustor
        self.turbine = turbine
        self.nozzle = nozzle
        self.air_mass_flow = air_mass_flow  # kg/s, physical mass flow AT THE DESIGN POINT

        # Reference (design) compressor-face conditions, used to scale mass
        # flow at off-design conditions under a constant-corrected-flow
        # assumption: mdot_corrected = mdot * sqrt(T2)/P2 = const for a
        # fixed-geometry engine running at a fixed corrected spool speed.
        # This is the standard first-order way to capture altitude/Mach
        # lapse rate without full compressor maps.
        self._design_condition = design_condition or FlightCondition(altitude=0.0, mach=0.0)
        design_freestream = self._design_condition.freestream_state(self.air_mass_flow)
        design_state_2 = self.inlet.process(design_freestream)
        self._design_p2 = design_state_2.total_pressure
        self._design_t2 = design_state_2.total_temperature

    def _scaled_mass_flow(self, state_2: GasState) -> float:
        """Physical mass flow at this operating point, holding corrected
        flow (mdot * sqrt(T2) / P2) constant relative to the design point."""
        corrected_flow_ratio = (
            (state_2.total_pressure / self._design_p2)
            * math.sqrt(self._design_t2 / state_2.total_temperature)
        )
        return self.air_mass_flow * corrected_flow_ratio

    def run_cycle(self, flight_condition: FlightCondition) -> PerformanceResult:
        # Station 0-2: freestream -> compressor face (mass flow is a first
        # guess here; corrected below once we know state_2's P/T).
        freestream = flight_condition.freestream_state(self.air_mass_flow)
        state_2 = self.inlet.process(freestream)

        # Re-scale mass flow based on this operating point's compressor-face
        # conditions (pressure/temperature don't depend on mass flow, so this
        # single correction pass is sufficient — no iteration needed).
        scaled_flow = self._scaled_mass_flow(state_2)
        state_2 = state_2.copy_with(mass_flow=scaled_flow)

        # Station 2-3: compressor
        state_3 = self.compressor.process(state_2)

        # Station 3-4: combustor
        state_4 = self.combustor.process(state_3)

        # Spool balance: turbine work must equal compressor work
        compressor_work = self.compressor.work_required(state_2, state_3)
        self.turbine.required_work = compressor_work

        # Station 4-5: turbine
        state_5 = self.turbine.process(state_4)

        # Station 5-9: nozzle expansion
        ambient_pressure = flight_condition.ambient_pressure()
        v_exit, p_exit, is_choked = self.nozzle.exit_conditions(state_5, ambient_pressure)

        # Thrust equation: F = mdot_exit * V_exit - mdot_inlet * V_flight
        #                     + (p_exit - p_ambient) * A_exit   [pressure thrust if not fully expanded]
        v_flight = flight_condition.flight_speed()
        momentum_thrust = state_5.mass_flow * v_exit - scaled_flow * v_flight

        if is_choked:
            # Approximate pressure thrust term for the choked case using the
            # exit area implied by continuity (A = mdot / (rho * V)).
            r_specific = state_5.cp * (state_5.gamma - 1) / state_5.gamma
            t_exit = state_5.total_temperature * (2 / (state_5.gamma + 1))
            rho_exit = p_exit / (r_specific * t_exit)
            area_exit = state_5.mass_flow / (rho_exit * v_exit)
            pressure_thrust = (p_exit - ambient_pressure) * area_exit
        else:
            pressure_thrust = 0.0

        thrust = momentum_thrust + pressure_thrust

        fuel_flow = scaled_flow * state_4.fuel_air_ratio
        tsfc = fuel_flow / thrust if thrust > 0 else float("inf")

        return PerformanceResult(
            thrust=thrust,
            tsfc=tsfc,
            fuel_flow=fuel_flow,
            exit_velocity=v_exit,
            is_choked=is_choked,
            turbine_inlet_temp=state_4.total_temperature,
            stations={
                "freestream": freestream,
                "state_2": state_2,
                "state_3": state_3,
                "state_4": state_4,
                "state_5": state_5,
            },
        )

    def get_calibratable_params(self) -> dict:
        """Returns current efficiency values — the parameters a Calibrator can tune."""
        return {
            "inlet_recovery": self.inlet.efficiency,
            "compressor_eff": self.compressor.efficiency,
            "combustor_eff": self.combustor.efficiency,
            "turbine_eff": self.turbine.efficiency,
            "nozzle_eff": self.nozzle.efficiency,
        }

    def set_calibratable_params(self, params: dict) -> None:
        """Setter used by a Calibrator to update efficiencies and re-run the cycle."""
        self.inlet.efficiency = params.get("inlet_recovery", self.inlet.efficiency)
        self.compressor.efficiency = params.get("compressor_eff", self.compressor.efficiency)
        self.combustor.efficiency = params.get("combustor_eff", self.combustor.efficiency)
        self.turbine.efficiency = params.get("turbine_eff", self.turbine.efficiency)
        self.nozzle.efficiency = params.get("nozzle_eff", self.nozzle.efficiency)
