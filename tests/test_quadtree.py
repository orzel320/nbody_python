import numpy as np
import pytest
from nbody_sim.quadtree import compute_bounding_box, build_tree

def test_compute_bounding_box():
    """Test that the bounding box perfectly encloses all bodies with a slight padding."""
    positions = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [0.0, 10.0],
        [10.0, 10.0]
    ], dtype=np.float64)

    cx, cy, size = compute_bounding_box(positions)

    # Center should be exactly in the middle
    assert cx == pytest.approx(5.0)
    assert cy == pytest.approx(5.0)

    # Size should be 10, plus the tiny 1.00001 padding we defined
    assert size == pytest.approx(10.0001)

def test_build_tree_root_properties():
    """Test that the root node correctly aggregates the mass and center of mass."""
    # Place 4 bodies perfectly in the 4 separate quadrants
    positions = np.array([
        [1.0, 1.0],   # NE quadrant
        [-1.0, 1.0],  # NW quadrant
        [-1.0, -1.0], # SW quadrant
        [1.0, -1.0]   # SE quadrant
    ], dtype=np.float64)
    
    masses = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    # In a Barnes-Hut tree, bodies get sorted in memory as they are placed into quadrants.
    # We pass an index array that the algorithm will shuffle around.
    body_indices = np.arange(4, dtype=np.int32)

    # Our build_tree function should return the filled array and how many nodes it used
    tree, num_nodes = build_tree(positions, masses, body_indices)

    # 1. The root node is always at index 0. It should contain the total system mass.
    assert tree[0]['mass'] == pytest.approx(10.0)

    # 2. Check the Center of Mass (CoM).
    # CoM X = (1*1 + -1*2 + -1*3 + 1*4) / 10 = 0.0
    # CoM Y = (1*1 + 1*2 + -1*3 + -1*4) / 10 = -0.4
    assert tree[0]['pos_x'] == pytest.approx(0.0)
    assert tree[0]['pos_y'] == pytest.approx(-0.4)

    # 3. The root node must account for all 4 bodies
    assert tree[0]['body_end'] - tree[0]['body_start'] == 4

    # 4. Because the bodies are far apart, the root must have split into branches.
    # -1 means leaf. Anything else is the index of its first child.
    assert tree[0]['children'] != -1

def test_build_tree_leaf_capacity():
    """Test that bodies close together stay in a single leaf node."""
    # Place 3 bodies extremely close together
    positions = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [-0.1, -0.1]
    ], dtype=np.float64)
    masses = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    body_indices = np.arange(3, dtype=np.int32)

    # Assuming we pass a leaf_capacity parameter (e.g., max bodies before splitting)
    tree, num_nodes = build_tree(positions, masses, body_indices, leaf_capacity=4)

    # Because there are only 3 bodies and capacity is 4, the root should not split.
    # It should be a leaf node.
    assert tree[0]['children'] == -1
    # Only 1 node (the root) should have been created
    assert num_nodes == 1