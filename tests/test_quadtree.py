import numpy as np
import pytest
from nbody_sim.quadtree import compute_bounding_box, build_tree

def test_compute_bounding_box():
    """Test that the bounding box perfectly encloses all bodies with a slight padding."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [10.0, 10.0, 10.0]
    ], dtype=np.float64)

    cx, cy, cz, size = compute_bounding_box(positions)

    assert cx == pytest.approx(5.0)
    assert cy == pytest.approx(5.0)
    assert cz == pytest.approx(5.0)
    assert size == pytest.approx(10.0001)

def test_build_tree_root_properties():
    """Test that the root node correctly aggregates the mass and center of mass."""
    positions = np.array([
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0]
    ], dtype=np.float64)
    
    masses = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    body_indices = np.arange(4, dtype=np.int32)

    tree, num_nodes = build_tree(positions, masses, body_indices)

    assert tree[0]['mass'] == pytest.approx(10.0)
    assert tree[0]['pos_x'] == pytest.approx(0.0)
    assert tree[0]['pos_y'] == pytest.approx(-0.4)
    assert tree[0]['pos_z'] == pytest.approx(-0.2)
    assert tree[0]['body_end'] - tree[0]['body_start'] == 4
    assert tree[0]['children'] != -1

def test_build_tree_leaf_capacity():
    """Test that bodies close together stay in a single leaf node."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [-0.1, -0.1, -0.1]
    ], dtype=np.float64)
    masses = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    body_indices = np.arange(3, dtype=np.int32)

    tree, num_nodes = build_tree(positions, masses, body_indices, leaf_capacity=4)

    assert tree[0]['children'] == -1
    assert num_nodes == 1