"""
Unit tests validating the physics model against known/expected behavior.

Run with: python3 -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from engine.atmosphere import StandardAtmosphere
from engine.components import GasState, Inlet, Compressor, Combustor, Turbine, Nozzle, CP_AIR, GAMMA_AIR
from engine.engine import TurbojetEngine, FlightCondition
from data.reference_engine import build_reference_engine, TRUE_EFFICIENCIES


class TestAtmosphere:
    def test_sea_level_matches_isa_standard(self):
        state = StandardAtmosphere.at_altitude(0)
        assert state.temperature == pytest.approx(288.15, abs=0.01)
        assert state.pressure == pytest.approx(101325, abs=1)
        assert state.speed_of_sound == pytest.approx(340.3, abs=0.2)

    def test_tropopause_pressure_matches_isa_table(self):
        # Published ISA value at 11 km is 22632 Pa
        state = StandardAtmosphere.at_altitude(11000)
        assert state.pressure == pytest.approx(22632, rel=0.001)

    def test_temperature_decreases_with_altitude_in_troposphere(self):
        low = StandardAtmosphere.at_altitude(1000)
        high = StandardAtmosphere.at_altitude(9000)
        assert high.temperature < low.temperature

    def test_rejects_altitude_above_model_range(self):
        with pytest.raises(ValueError):
            StandardAtmosphere.at_altitude(25000)


class TestComponents:
    def test_compressor_raises_pressure_by_design_ratio(self):
        compressor = Compressor(efficiency=0.85, pressure_ratio=7.0)
        inlet_state = GasState(total_pressure=101325, total_temperature=288.0, mass_flow=20.0)
        outlet = compressor.process(inlet_state)
        assert outlet.total_pressure == pytest.approx(101325 * 7.0)

    def test_compressor_lower_efficiency_means_higher_exit_temp(self):
        # Physical sanity check: a less efficient compressor does MORE
        # irreversible work for the same pressure rise -> hotter exit air.
        inlet_state = GasState(total_pressure=101325, total_temperature=288.0, mass_flow=20.0)
        efficient = Compressor(efficiency=0.90, pressure_ratio=7.0).process(inlet_state)
        inefficient = Compressor(efficiency=0.75, pressure_ratio=7.0).process(inlet_state)
        assert inefficient.total_temperature > efficient.total_temperature

    def test_combustor_raises_temperature_to_target_tit(self):
        combustor = Combustor(efficiency=0.99, fuel_heating_value=43e6, exit_temperature=1200.0)
        inlet_state = GasState(total_pressure=700000, total_temperature=600.0, mass_flow=20.0)
        outlet = combustor.process(inlet_state)
        assert outlet.total_temperature == pytest.approx(1200.0)
        assert outlet.fuel_air_ratio > 0

    def test_combustor_applies_pressure_loss(self):
        combustor = Combustor(efficiency=0.99, fuel_heating_value=43e6,
                               exit_temperature=1200.0, pressure_loss_fraction=0.04)
        inlet_state = GasState(total_pressure=700000, total_temperature=600.0, mass_flow=20.0)
        outlet = combustor.process(inlet_state)
        assert outlet.total_pressure == pytest.approx(700000 * 0.96)

    def test_turbine_extracts_exactly_the_required_work(self):
        turbine = Turbine(efficiency=0.90)
        turbine.required_work = 5.0e6  # W
        inlet_state = GasState(total_pressure=400000, total_temperature=1200.0,
                                mass_flow=20.4, cp=1244.0, gamma=1.333)
        outlet = turbine.process(inlet_state)
        actual_work_extracted = inlet_state.mass_flow * inlet_state.cp * (
            inlet_state.total_temperature - outlet.total_temperature)
        assert actual_work_extracted == pytest.approx(5.0e6, rel=1e-6)

    def test_efficiency_bounds_are_enforced(self):
        with pytest.raises(ValueError):
            Compressor(efficiency=1.5, pressure_ratio=7.0)
        with pytest.raises(ValueError):
            Compressor(efficiency=0.0, pressure_ratio=7.0)


class TestEngineIntegration:
    def test_reference_engine_produces_positive_thrust(self):
        engine = build_reference_engine()
        result = engine.run_cycle(FlightCondition(altitude=0, mach=0.0))
        assert result.thrust > 0

    def test_reference_engine_sea_level_static_thrust_is_realistic(self):
        # Small military turbojets (J85-class) are ~13-16 kN SLS thrust.
        engine = build_reference_engine()
        result = engine.run_cycle(FlightCondition(altitude=0, mach=0.0))
        assert 10_000 < result.thrust < 20_000

    def test_thrust_decreases_substantially_at_cruise_altitude(self):
        # Physical expectation: cruise-altitude thrust should be roughly
        # 1/4 to 1/2 of sea-level-static thrust for a pure turbojet.
        engine = build_reference_engine()
        sls = engine.run_cycle(FlightCondition(altitude=0, mach=0.0))
        cruise = engine.run_cycle(FlightCondition(altitude=11000, mach=0.8))
        ratio = cruise.thrust / sls.thrust
        assert 0.2 < ratio < 0.6

    def test_degraded_compressor_efficiency_reduces_thrust(self):
        healthy = build_reference_engine()
        degraded_effs = dict(TRUE_EFFICIENCIES)
        degraded_effs["compressor_eff"] = 0.70
        degraded = build_reference_engine(degraded_effs)

        cond = FlightCondition(altitude=0, mach=0.0)
        healthy_result = healthy.run_cycle(cond)
        degraded_result = degraded.run_cycle(cond)
        assert degraded_result.thrust < healthy_result.thrust

    def test_get_and_set_calibratable_params_roundtrip(self):
        engine = build_reference_engine()
        original = engine.get_calibratable_params()
        modified = dict(original)
        modified["compressor_eff"] = 0.77
        engine.set_calibratable_params(modified)
        assert engine.get_calibratable_params()["compressor_eff"] == pytest.approx(0.77)
        # restore
        engine.set_calibratable_params(original)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
