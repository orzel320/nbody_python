import pytest
import numpy as np
import pint
from nbody_sim.api import Simulation, ureg

def test_simulation_strips_units():
    """Test that the API accepts N-body units and strips them directly to floats."""
    masses = [1.0, 5.0] * ureg.solar_mass
    positions = [
        [0.0, 0.0],
        [1.5, 2.0]
    ] * ureg.astronomical_unit
    velocities = [
        [10.0, 0.0],
        [0.0, -5.0]
    ] * (ureg.astronomical_unit / ureg.nbody_time)
    
    sim = Simulation()
    sim.add_bodies(positions, velocities, masses)
    
    assert isinstance(sim._positions, np.ndarray)
    assert not isinstance(sim._positions, pint.Quantity)
    
    assert sim._masses[0] == 1.0
    assert sim._positions[1][0] == 1.5
    assert sim._velocities[0][0] == 10.0

def test_simulation_unit_conversion():
    """Test that the API correctly scales SI/Metric units to N-body units."""
    
    masses = [5.972e24] * ureg.kilogram
    positions = [[1.495978707e11, 0.0]] * ureg.meter
    velocities = [[30.0, 0.0]] * (ureg.kilometer / ureg.second)
    
    sim = Simulation()
    sim.add_bodies(positions, velocities, masses)
    
    expected_mass = (5.972e24 * ureg.kilogram).to(ureg.solar_mass).magnitude
    expected_pos = (1.495978707e11 * ureg.meter).to(ureg.astronomical_unit).magnitude
    expected_vel = (30.0 * ureg.kilometer / ureg.second).to(ureg.astronomical_unit / ureg.nbody_time).magnitude
    
    assert sim._masses[0] == pytest.approx(expected_mass)
    assert sim._positions[0][0] == pytest.approx(expected_pos)
    assert sim._velocities[0][0] == pytest.approx(expected_vel)