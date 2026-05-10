import numpy as np
import pytest
from nbody_sim.physics import calculate_acceleration, update_bodies, calculate_barnes_hut_accel
from nbody_sim.quadtree import build_tree
from nbody_sim.api import Simulation

def test_acceleration_basic():
    """Test standard gravitational pull between two bodies."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0]
    ])
    masses = np.array([100.0, 1.0])
    
    acc = calculate_acceleration(positions, masses, epsilon_sq=0.0)
    
    assert acc[1][0] == pytest.approx(-1.0)
    assert acc[1][1] == pytest.approx(0.0)
    assert acc[1][2] == pytest.approx(0.0)

def test_acceleration_softening():
    """Test that the softening parameter prevents infinite acceleration."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    masses = np.array([10.0, 10.0])
    
    acc = calculate_acceleration(positions, masses, epsilon_sq=1.0)
    
    assert not np.isnan(acc).any()
    assert not np.isinf(acc).any()
    assert np.all(acc == 0.0)

def test_acceleration_symmetry():
    """Newton's Third Law: Forces should be equal and opposite."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 4.0, 5.0]
    ])
    masses = np.array([10.0, 10.0])
    
    acc = calculate_acceleration(positions, masses, epsilon_sq=0.0)
    
    assert acc[0][0] == pytest.approx(-acc[1][0])
    assert acc[0][1] == pytest.approx(-acc[1][1])
    assert acc[0][2] == pytest.approx(-acc[1][2])

def test_symplectic_euler_integration():
    """Test that velocity and position update correctly over a time step."""
    dt = 0.5
    
    positions = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0]
    ])
    velocities = np.array([
        [2.0, 0.0, 0.0],
        [0.0, -4.0, 0.0]
    ])
    accelerations = np.array([
        [0.0, 2.0, 0.0],
        [-2.0, 0.0, 0.0]
    ])
    
    update_bodies(positions, velocities, accelerations, dt)
    
    assert velocities[0][0] == pytest.approx(2.0)
    assert velocities[0][1] == pytest.approx(1.0)
    assert positions[0][0] == pytest.approx(1.0)
    assert positions[0][1] == pytest.approx(0.5)

    assert velocities[1][0] == pytest.approx(-1.0)
    assert velocities[1][1] == pytest.approx(-4.0)
    assert positions[1][0] == pytest.approx(9.5)
    assert positions[1][1] == pytest.approx(8.0)

def test_barnes_hut_matches_exact():
    """Test that Barnes-Hut (with theta=0) matches the exact O(N^2) calculation."""
    np.random.seed(42)
    n = 50
    positions = np.random.uniform(-10.0, 10.0, (n, 3)).astype(np.float64)
    masses = np.random.uniform(1.0, 10.0, n).astype(np.float64)
    body_indices = np.arange(n, dtype=np.int32)
    
    acc_exact = calculate_acceleration(positions, masses, epsilon_sq=0.01)
    
    tree, _ = build_tree(positions, masses, body_indices, leaf_capacity=1)
    
    acc_bh = calculate_barnes_hut_accel(
        positions, masses, tree, body_indices, theta_sq=0.0, epsilon_sq=0.01
    )
    
    np.testing.assert_allclose(acc_bh, acc_exact, rtol=1e-5, atol=1e-5)
