"""
Turbojet engine components.

Each component implements isentropic-relation based thermodynamics with a
component efficiency that captures real-world losses. This follows the
standard "cold air standard cycle analysis" approach used in introductory
gas turbine textbooks (e.g. Mattingly, "Elements of Gas Turbine Propulsion").

Simplifications made (typical for a first-pass performance model):
- Calorically perfect gas (constant cp, gamma) on each side of the
  combustor (cold-side air properties vs hot-side combustion gas properties).
- No compressor bleed air, no variable geometry, no afterburner.
- Nozzle assumed fully expanded (exit pressure = ambient) unless choked.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import math

# Gas properties (cold air standard assumptions)
CP_AIR = 1004.5      # J/(kg*K), specific heat of air (cold side)
GAMMA_AIR = 1.4       # ratio of specific heats, cold side
CP_GAS = 1244.0       # J/(kg*K), specific heat of combustion gas (hot side)
GAMMA_GAS = 1.333     # ratio of specific heats, hot side


@dataclass
class GasState:
    """Thermodynamic state (stagnation/total conditions) at an engine station."""
    total_pressure: float      # Pa
    total_temperature: float   # K
    mass_flow: float           # kg/s
    fuel_air_ratio: float = 0.0
    cp: float = CP_AIR
    gamma: float = GAMMA_AIR

    def copy_with(self, **kwargs) -> "GasState":
        return replace(self, **kwargs)


class EngineComponent(ABC):
    """Abstract base class for any component in the gas path."""

    def __init__(self, efficiency: float, name: str = ""):
        if not (0.0 < efficiency <= 1.0):
            raise ValueError(f"Efficiency must be in (0, 1], got {efficiency}")
        self.efficiency = efficiency
        self.name = name or self.__class__.__name__

    @abstractmethod
    def process(self, inlet: GasState) -> GasState:
        """Transform the inlet gas state into the outlet gas state."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}(efficiency={self.efficiency:.3f})"


class Inlet(EngineComponent):
    """
    Diffuser: decelerates freestream flow to near-zero velocity at the
    compressor face, converting kinetic energy to a pressure/temperature
    rise (ram effect). `efficiency` here is the ram pressure recovery factor.
    """

    def process(self, inlet: GasState) -> GasState:
        # inlet.total_pressure/total_temperature are ALREADY freestream
        # stagnation values (computed via FlightCondition). The inlet
        # component only applies a recovery loss to stagnation pressure;
        # stagnation temperature is conserved (no work/heat added).
        recovered_pressure = inlet.total_pressure * self.efficiency
        return inlet.copy_with(total_pressure=recovered_pressure)


class Compressor(EngineComponent):
    """
    Raises stagnation pressure by a design pressure ratio. `efficiency` is
    the isentropic efficiency: ratio of ideal-to-actual temperature rise
    for the same pressure ratio.
    """

    def __init__(self, efficiency: float, pressure_ratio: float, name: str = ""):
        super().__init__(efficiency, name)
        self.pressure_ratio = pressure_ratio

    def process(self, inlet: GasState) -> GasState:
        gamma = inlet.gamma
        p_out = inlet.total_pressure * self.pressure_ratio

        # Ideal (isentropic) exit temperature for this pressure ratio
        t_out_ideal = inlet.total_temperature * self.pressure_ratio ** ((gamma - 1) / gamma)
        ideal_temp_rise = t_out_ideal - inlet.total_temperature

        # Isentropic efficiency: eta_c = ideal_rise / actual_rise
        actual_temp_rise = ideal_temp_rise / self.efficiency
        t_out = inlet.total_temperature + actual_temp_rise

        return inlet.copy_with(total_pressure=p_out, total_temperature=t_out)

    def work_required(self, inlet: GasState, outlet: GasState) -> float:
        """Shaft power (W) needed to drive this compressor."""
        return inlet.mass_flow * inlet.cp * (outlet.total_temperature - inlet.total_temperature)


class Combustor(EngineComponent):
    """
    Adds heat via fuel combustion at (approximately) constant pressure.
    `efficiency` is combustion efficiency (fraction of fuel energy that
    actually goes into raising gas temperature).
    """

    def __init__(self, efficiency: float, fuel_heating_value: float,
                 exit_temperature: float, pressure_loss_fraction: float = 0.04,
                 name: str = ""):
        super().__init__(efficiency, name)
        self.fuel_heating_value = fuel_heating_value    # J/kg, e.g. Jet-A ~ 43e6
        self.exit_temperature = exit_temperature         # K, turbine inlet temp (TIT) - design limit
        self.pressure_loss_fraction = pressure_loss_fraction

    def process(self, inlet: GasState) -> GasState:
        t_in = inlet.total_temperature
        t_out = self.exit_temperature

        # Energy balance to solve for fuel-air ratio f:
        #   mdot_fuel * eta_b * Q_R = mdot_gas_out * cp_gas * T_out - mdot_air * cp_air * T_in
        # with mdot_fuel = f * mdot_air, mdot_gas_out = mdot_air * (1 + f)
        # Rearranged (standard combustor sizing formula):
        f = (CP_GAS * t_out - CP_AIR * t_in) / (self.efficiency * self.fuel_heating_value - CP_GAS * t_out)

        if f <= 0:
            raise ValueError(
                f"Computed fuel-air ratio is non-physical ({f:.4f}). "
                f"Check exit_temperature ({t_out}K) vs inlet ({t_in:.1f}K)."
            )

        p_out = inlet.total_pressure * (1 - self.pressure_loss_fraction)
        mass_flow_out = inlet.mass_flow * (1 + f)

        return GasState(
            total_pressure=p_out,
            total_temperature=t_out,
            mass_flow=mass_flow_out,
            fuel_air_ratio=f,
            cp=CP_GAS,
            gamma=GAMMA_GAS,
        )


class Turbine(EngineComponent):
    """
    Extracts just enough work from the hot gas stream to drive the
    compressor (single-spool assumption). `efficiency` is isentropic
    turbine efficiency.
    """

    def __init__(self, efficiency: float, name: str = ""):
        super().__init__(efficiency, name)
        self.required_work: float = 0.0  # W, set externally by the Engine (spool balance)

    def process(self, inlet: GasState) -> GasState:
        gamma = inlet.gamma
        cp = inlet.cp

        # Actual temperature drop needed to supply required shaft work
        actual_temp_drop = self.required_work / (inlet.mass_flow * cp)
        t_out = inlet.total_temperature - actual_temp_drop

        # Ideal (isentropic) temp drop would be smaller in magnitude by eta_t
        # eta_t = actual_drop / ideal_drop  =>  ideal_drop = actual_drop / eta_t
        ideal_temp_drop = actual_temp_drop / self.efficiency
        t_out_ideal = inlet.total_temperature - ideal_temp_drop

        # From the ideal temperature ratio, back out the pressure ratio
        pressure_ratio = (t_out_ideal / inlet.total_temperature) ** (gamma / (gamma - 1))
        p_out = inlet.total_pressure * pressure_ratio

        return inlet.copy_with(total_pressure=p_out, total_temperature=t_out)


class Nozzle(EngineComponent):
    """
    Expands the hot gas to (or towards) ambient pressure, converting
    pressure/thermal energy into kinetic energy (thrust). `efficiency` is
    the nozzle's isentropic/velocity coefficient.
    """

    def process(self, inlet: GasState) -> GasState:
        # process() just passes the state through; actual expansion physics
        # live in exit_velocity() since it also needs ambient pressure.
        return inlet

    def exit_conditions(self, inlet: GasState, ambient_pressure: float):
        """
        Returns (exit_velocity, exit_pressure, is_choked).
        Handles the choked-flow case (common for jet engines at high power).
        """
        gamma = inlet.gamma
        cp = inlet.cp

        critical_pressure_ratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        pressure_ratio_available = ambient_pressure / inlet.total_pressure

        if pressure_ratio_available <= critical_pressure_ratio:
            # Choked: nozzle exit is at sonic conditions (M=1), exit
            # pressure is above ambient (not fully expanded).
            is_choked = True
            t_exit = inlet.total_temperature * (2 / (gamma + 1))
            r_specific = cp * (gamma - 1) / gamma
            v_exit = math.sqrt(gamma * r_specific * t_exit)
            p_exit = inlet.total_pressure * critical_pressure_ratio
        else:
            # Fully expanded to ambient pressure
            is_choked = False
            p_exit = ambient_pressure
            temp_ratio_ideal = pressure_ratio_available ** ((gamma - 1) / gamma)
            t_exit_ideal = inlet.total_temperature * temp_ratio_ideal
            ideal_ke_drop = cp * (inlet.total_temperature - t_exit_ideal)
            actual_ke_drop = ideal_ke_drop * self.efficiency  # nozzle velocity coefficient
            v_exit = math.sqrt(max(2 * actual_ke_drop, 0.0))

        return v_exit, p_exit, is_choked
