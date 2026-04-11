import numpy as np
from numba import njit

# This defines the exact memory layout of a single Node in our Quadtree.
# It perfectly mirrors the Rust `Node` struct.
node_dtype = np.dtype([
    ('mass', np.float64),
    ('pos_x', np.float64),        # Center of mass X
    ('pos_y', np.float64),        # Center of mass Y
    ('center_x', np.float64),     # Geometric center of the quad
    ('center_y', np.float64),     # Geometric center of the quad
    ('size', np.float64),         # Width/Height of the quad
    ('children', np.int32),       # Index of the first child (-1 if leaf)
    ('body_start', np.int32),     # Starting index of bodies in this node
    ('body_end', np.int32),       # Ending index of bodies in this node
])

@njit
def create_empty_tree(max_nodes):
    """Allocates the memory for the entire tree upfront."""
    tree = np.zeros(max_nodes, dtype=node_dtype)
    # Numba requires explicit loops for structured array assignment
    for i in range(max_nodes):
        tree[i]['children'] = -1 
    return tree

@njit(fastmath=True)
def compute_bounding_box(positions):
    """Finds the center and size of a square that contains all bodies."""
    n = positions.shape[0]
    
    # Find min and max bounds
    min_x = positions[0, 0]
    max_x = positions[0, 0]
    min_y = positions[0, 1]
    max_y = positions[0, 1]
    
    for i in range(1, n):
        if positions[i, 0] < min_x: min_x = positions[i, 0]
        if positions[i, 0] > max_x: max_x = positions[i, 0]
        if positions[i, 1] < min_y: min_y = positions[i, 1]
        if positions[i, 1] > max_y: max_y = positions[i, 1]
        
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    
    # The size must be a square, so we take the maximum dimension
    width = max_x - min_x
    height = max_y - min_y
    size = max(width, height)
    
    # Add a tiny bit of padding to prevent floating point edge cases
    # where a body sits exactly on the boundary
    size *= 1.00001 
    
    return center_x, center_y, size

@njit(fastmath=True)
def partition(indices, positions, start, end, axis, pivot):
    """
    In-place partitioning of the body indices array.
    Groups bodies that are < pivot to the left, and >= pivot to the right.
    Returns the index where the split occurs.
    """
    i = start
    j = end - 1
    while True:
        while i <= j and positions[indices[i], axis] < pivot:
            i += 1
        while i <= j and positions[indices[j], axis] >= pivot:
            j -= 1
        if i >= j:
            return i
        
        # Swap indices
        tmp = indices[i]
        indices[i] = indices[j]
        indices[j] = tmp
        i += 1
        j -= 1

@njit(fastmath=True)
def build_tree(positions, masses, body_indices, leaf_capacity=1):
    """
    Builds the Quadtree using an explicit stack and flat memory arrays.
    """
    n = positions.shape[0]
    
    # Pre-allocate memory for the tree. A safe upper bound is 4 * N nodes.
    max_nodes = (4 * n) + 1024
    tree = create_empty_tree(max_nodes)
    
    # We store the indices of branch nodes so we can calculate 
    # the center of mass bottom-up later.
    parents = np.zeros(max_nodes // 4, dtype=np.int32)
    
    cx, cy, size = compute_bounding_box(positions)

    # Initialize the Root Node
    tree[0]['center_x'] = cx
    tree[0]['center_y'] = cy
    tree[0]['size'] = size
    tree[0]['body_start'] = 0
    tree[0]['body_end'] = n

    node_counter = 1
    parent_counter = 0

    # Explicit stack for top-down construction (stores node indices)
    stack = np.zeros(max_nodes, dtype=np.int32)
    stack_ptr = 0
    
    # Push root to stack
    stack[stack_ptr] = 0
    stack_ptr += 1

    # 1. TOP-DOWN SUBDIVISION
    while stack_ptr > 0:
        stack_ptr -= 1
        node = stack[stack_ptr]

        start = tree[node]['body_start']
        end = tree[node]['body_end']

        # LEAF NODE CONDITION
        if end - start <= leaf_capacity:
            total_mass = 0.0
            cm_x = 0.0
            cm_y = 0.0
            for i in range(start, end):
                idx = body_indices[i]
                m = masses[idx]
                total_mass += m
                cm_x += positions[idx, 0] * m
                cm_y += positions[idx, 1] * m
                
            tree[node]['mass'] = total_mass
            if total_mass > 0:
                tree[node]['pos_x'] = cm_x / total_mass
                tree[node]['pos_y'] = cm_y / total_mass
            continue

        # BRANCH NODE (Subdivide into 4 quadrants)
        c_x = tree[node]['center_x']
        c_y = tree[node]['center_y']
        half_size = tree[node]['size'] * 0.5
        quarter_size = half_size * 0.5

        # Partition the bodies into SW, SE, NW, NE
        # Axis 1 is Y, Axis 0 is X
        mid_y = partition(body_indices, positions, start, end, 1, c_y)
        mid_x1 = partition(body_indices, positions, start, mid_y, 0, c_x)
        mid_x2 = partition(body_indices, positions, mid_y, end, 0, c_x)

        splits = [start, mid_x1, mid_y, mid_x2, end]

        children_start = node_counter
        tree[node]['children'] = children_start
        node_counter += 4

        parents[parent_counter] = node
        parent_counter += 1

        # Calculate geometric centers for the 4 new children
        offsets_x = [-quarter_size, quarter_size, -quarter_size, quarter_size]
        offsets_y = [-quarter_size, -quarter_size, quarter_size, quarter_size]

        for i in range(4):
            child = children_start + i
            tree[child]['center_x'] = c_x + offsets_x[i]
            tree[child]['center_y'] = c_y + offsets_y[i]
            tree[child]['size'] = half_size
            tree[child]['body_start'] = splits[i]
            tree[child]['body_end'] = splits[i+1]

            # If the quadrant has bodies, push it to the stack to be processed
            if splits[i+1] > splits[i]:
                stack[stack_ptr] = child
                stack_ptr += 1

    # 2. BOTTOM-UP PROPAGATION (Center of Mass)
    # We iterate backwards through the parents array. This guarantees that 
    # we process deeper branches before their parents.
    for p_idx in range(parent_counter - 1, -1, -1):
        node = parents[p_idx]
        first_child = tree[node]['children']

        total_mass = 0.0
        cm_x = 0.0
        cm_y = 0.0

        for i in range(4):
            child = first_child + i
            m = tree[child]['mass']
            total_mass += m
            cm_x += tree[child]['pos_x'] * m
            cm_y += tree[child]['pos_y'] * m

        tree[node]['mass'] = total_mass
        if total_mass > 0:
            tree[node]['pos_x'] = cm_x / total_mass
            tree[node]['pos_y'] = cm_y / total_mass

    return tree, node_counter