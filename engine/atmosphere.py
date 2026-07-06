"""
International Standard Atmosphere (ISA) model.

Implements the standard atmosphere up to 20 km (covers troposphere +
lower stratosphere, which is where essentially all air-breathing engine
performance analysis happens).

Reference: ICAO Standard Atmosphere, 1993.
"""

from dataclasses import dataclass

# Sea-level reference conditions
T0_SL = 288.15       # K
P0_SL = 101325.0     # Pa
RHO0_SL = 1.225      # kg/m^3

R_AIR = 287.05       # J/(kg*K), specific gas constant for air
G0 = 9.80665         # m/s^2

# Tropopause (11 km) reference values
T_TROPOPAUSE = 216.65   # K, constant above 11 km up to 20 km
H_TROPOPAUSE = 11000.0  # m
LAPSE_RATE = -0.0065    # K/m, troposphere temperature lapse rate

# Pressure at the tropopause (computed from the barometric formula)
P_TROPOPAUSE = P0_SL * (T_TROPOPAUSE / T0_SL) ** (-G0 / (LAPSE_RATE * R_AIR))


@dataclass
class AtmosphereState:
    """Ambient conditions at a given altitude."""
    altitude: float          # m
    temperature: float       # K (static)
    pressure: float          # Pa (static)
    density: float           # kg/m^3
    speed_of_sound: float    # m/s

    def __repr__(self):
        return (f"AtmosphereState(alt={self.altitude:.0f}m, "
                f"T={self.temperature:.1f}K, P={self.pressure:.0f}Pa, "
                f"a={self.speed_of_sound:.1f}m/s)")


class StandardAtmosphere:
    """Computes ISA static conditions as a function of geometric altitude."""

    GAMMA_AIR = 1.4  # ratio of specific heats for cold air

    @classmethod
    def at_altitude(cls, altitude_m: float) -> AtmosphereState:
        if altitude_m < 0:
            raise ValueError("Altitude must be non-negative.")
        if altitude_m > 20000:
            raise ValueError("This simplified model only covers 0-20 km "
                              "(troposphere + lower stratosphere).")

        if altitude_m <= H_TROPOPAUSE:
            temperature = T0_SL + LAPSE_RATE * altitude_m
            pressure = P0_SL * (temperature / T0_SL) ** (-G0 / (LAPSE_RATE * R_AIR))
        else:
            # Isothermal layer: 11 km - 20 km
            temperature = T_TROPOPAUSE
            pressure = P_TROPOPAUSE * pow(
                2.718281828459045,
                -G0 * (altitude_m - H_TROPOPAUSE) / (R_AIR * T_TROPOPAUSE)
            )

        density = pressure / (R_AIR * temperature)
        speed_of_sound = (cls.GAMMA_AIR * R_AIR * temperature) ** 0.5

        return AtmosphereState(
            altitude=altitude_m,
            temperature=temperature,
            pressure=pressure,
            density=density,
            speed_of_sound=speed_of_sound,
        )
