import numpy as np
import pytest
from nbody_sim.physics import calculate_acceleration

def test_acceleration_basic():
    """Test standard gravitational pull between two bodies."""
    # Body 0: Origin, Mass 100
    # Body 1: x=10, y=0, Mass 1
    positions = np.array([
        [0.0, 0.0],
        [10.0, 0.0]
    ])
    masses = np.array([100.0, 1.0])
    
    # Using a softening parameter (epsilon) of 0.0 for pure math
    acc = calculate_acceleration(positions, masses, epsilon_sq=0.0)
    
    # Body 1 should be pulled to the left (negative X) by Body 0.
    # Force = G * (m1 * m2) / r^2.  Acceleration = Force / m2.
    # We are using G=1 for this simulation.
    # a = 1 * 100 / 100 = 1.0. Vector should be [-1.0, 0.0]
    
    assert acc[1][0] == pytest.approx(-1.0)
    assert acc[1][1] == pytest.approx(0.0)

def test_acceleration_softening():
    """Test that the softening parameter prevents infinite acceleration."""
    # Two bodies in the exact same position
    positions = np.array([
        [0.0, 0.0],
        [0.0, 0.0]
    ])
    masses = np.array([10.0, 10.0])
    
    # With epsilon > 0, acceleration should be 0 (they cancel out), but NOT NaN/Inf
    acc = calculate_acceleration(positions, masses, epsilon_sq=1.0)
    
    assert not np.isnan(acc).any()
    assert not np.isinf(acc).any()
    assert np.all(acc == 0.0)

def test_acceleration_symmetry():
    """Newton's Third Law: Forces should be equal and opposite."""
    positions = np.array([
        [0.0, 0.0],
        [3.0, 4.0] # 3-4-5 triangle, distance = 5
    ])
    masses = np.array([10.0, 10.0])
    
    acc = calculate_acceleration(positions, masses, epsilon_sq=0.0)
    
    # Because masses are equal, accelerations must be exactly opposite
    assert acc[0][0] == pytest.approx(-acc[1][0])
    assert acc[0][1] == pytest.approx(-acc[1][1])

from nbody_sim.physics import update_bodies

def test_symplectic_euler_integration():
    """Test that velocity and position update correctly over a time step."""
    dt = 0.5
    
    positions = np.array([
        [0.0, 0.0],
        [10.0, 10.0]
    ])
    velocities = np.array([
        [2.0, 0.0],  # Moving right
        [0.0, -4.0]  # Moving down
    ])
    accelerations = np.array([
        [0.0, 2.0],  # Accelerating up
        [-2.0, 0.0]  # Accelerating left
    ])
    
    # Update bodies in place
    update_bodies(positions, velocities, accelerations, dt)
    
    # Check Body 0 (starts at 0,0 | v=2,0 | a=0,2)
    # v_new_x = 2.0 + 0 * 0.5 = 2.0
    # v_new_y = 0.0 + 2 * 0.5 = 1.0
    # p_new_x = 0.0 + 2.0 * 0.5 = 1.0
    # p_new_y = 0.0 + 1.0 * 0.5 = 0.5
    assert velocities[0][0] == pytest.approx(2.0)
    assert velocities[0][1] == pytest.approx(1.0)
    assert positions[0][0] == pytest.approx(1.0)
    assert positions[0][1] == pytest.approx(0.5)

    # Check Body 1 (starts at 10,10 | v=0,-4 | a=-2,0)
    # v_new_x = 0.0 + (-2 * 0.5) = -1.0
    # v_new_y = -4.0 + 0 = -4.0
    # p_new_x = 10.0 + (-1.0 * 0.5) = 9.5
    # p_new_y = 10.0 + (-4.0 * 0.5) = 8.0
    assert velocities[1][0] == pytest.approx(-1.0)
    assert velocities[1][1] == pytest.approx(-4.0)
    assert positions[1][0] == pytest.approx(9.5)
    assert positions[1][1] == pytest.approx(8.0)