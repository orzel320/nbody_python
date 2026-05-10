import numpy as np
from numba import njit

import numpy as np
from numba import njit

node_dtype = np.dtype([
    ('mass', np.float64),
    ('pos_x', np.float64),
    ('pos_y', np.float64),
    ('pos_z', np.float64),
    ('center_x', np.float64),
    ('center_y', np.float64),
    ('center_z', np.float64),
    ('size', np.float64),
    ('children', np.int32),
    ('body_start', np.int32),
    ('body_end', np.int32),
])

@njit
def create_empty_tree(max_nodes):
    tree = np.zeros(max_nodes, dtype=node_dtype)
    for i in range(max_nodes):
        tree[i]['children'] = -1 
    return tree

@njit(fastmath=True)
def compute_bounding_box(positions):
    """Finds the center and size of a perfect cube that contains all bodies."""
    n = positions.shape[0]
    
    min_x = positions[0, 0]
    max_x = positions[0, 0]
    min_y = positions[0, 1]
    max_y = positions[0, 1]
    min_z = positions[0, 2]
    max_z = positions[0, 2]
    
    for i in range(1, n):
        if positions[i, 0] < min_x: min_x = positions[i, 0]
        if positions[i, 0] > max_x: max_x = positions[i, 0]
        if positions[i, 1] < min_y: min_y = positions[i, 1]
        if positions[i, 1] > max_y: max_y = positions[i, 1]
        if positions[i, 2] < min_z: min_z = positions[i, 2]
        if positions[i, 2] > max_z: max_z = positions[i, 2]
        
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    center_z = (min_z + max_z) * 0.5
    
    width = max_x - min_x
    height = max_y - min_y
    depth = max_z - min_z
    
    size = max(width, max(height, depth)) * 1.00001 
    
    return center_x, center_y, center_z, size

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
        
        tmp = indices[i]
        indices[i] = indices[j]
        indices[j] = tmp
        i += 1
        j -= 1

@njit(fastmath=True)
def build_tree(positions, masses, body_indices, leaf_capacity=1):
    """
    Builds the 3D Octree using an explicit stack and flat memory arrays.
    """
    n = positions.shape[0]
    
    max_nodes = (8 * n) + 1024
    tree = create_empty_tree(max_nodes)
    
    parents = np.zeros(max_nodes // 8, dtype=np.int32)
    
    cx, cy, cz, size = compute_bounding_box(positions)

    tree[0]['center_x'] = cx
    tree[0]['center_y'] = cy
    tree[0]['center_z'] = cz
    tree[0]['size'] = size
    tree[0]['body_start'] = 0
    tree[0]['body_end'] = n

    node_counter = 1
    parent_counter = 0

    stack = np.zeros(max_nodes, dtype=np.int32)
    stack_ptr = 0
    
    stack[stack_ptr] = 0
    stack_ptr += 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node = stack[stack_ptr]

        start = tree[node]['body_start']
        end = tree[node]['body_end']

        if end - start <= leaf_capacity:
            total_mass = 0.0
            cm_x = 0.0
            cm_y = 0.0
            cm_z = 0.0
            for i in range(start, end):
                idx = body_indices[i]
                m = masses[idx]
                total_mass += m
                cm_x += positions[idx, 0] * m
                cm_y += positions[idx, 1] * m
                cm_z += positions[idx, 2] * m
                
            tree[node]['mass'] = total_mass
            if total_mass > 0:
                tree[node]['pos_x'] = cm_x / total_mass
                tree[node]['pos_y'] = cm_y / total_mass
                tree[node]['pos_z'] = cm_z / total_mass
            continue

        c_x = tree[node]['center_x']
        c_y = tree[node]['center_y']
        c_z = tree[node]['center_z']
        half_size = tree[node]['size'] * 0.5
        quarter_size = half_size * 0.5

        mid_z = partition(body_indices, positions, start, end, 2, c_z)
        
        mid_y1 = partition(body_indices, positions, start, mid_z, 1, c_y)
        mid_y2 = partition(body_indices, positions, mid_z, end, 1, c_y)
        
        mid_x1 = partition(body_indices, positions, start, mid_y1, 0, c_x)
        mid_x2 = partition(body_indices, positions, mid_y1, mid_z, 0, c_x)
        mid_x3 = partition(body_indices, positions, mid_z, mid_y2, 0, c_x)
        mid_x4 = partition(body_indices, positions, mid_y2, end, 0, c_x)

        splits = [start, mid_x1, mid_y1, mid_x2, mid_z, mid_x3, mid_y2, mid_x4, end]

        children_start = node_counter
        tree[node]['children'] = children_start
        node_counter += 8

        parents[parent_counter] = node
        parent_counter += 1

        offsets_x = [-quarter_size, quarter_size, -quarter_size, quarter_size, -quarter_size, quarter_size, -quarter_size, quarter_size]
        offsets_y = [-quarter_size, -quarter_size, quarter_size, quarter_size, -quarter_size, -quarter_size, quarter_size, quarter_size]
        offsets_z = [-quarter_size, -quarter_size, -quarter_size, -quarter_size, quarter_size, quarter_size, quarter_size, quarter_size]

        for i in range(8):
            child = children_start + i
            tree[child]['center_x'] = c_x + offsets_x[i]
            tree[child]['center_y'] = c_y + offsets_y[i]
            tree[child]['center_z'] = c_z + offsets_z[i]
            tree[child]['size'] = half_size
            tree[child]['body_start'] = splits[i]
            tree[child]['body_end'] = splits[i+1]

            if splits[i+1] > splits[i]:
                stack[stack_ptr] = child
                stack_ptr += 1

    for p_idx in range(parent_counter - 1, -1, -1):
        node = parents[p_idx]
        first_child = tree[node]['children']

        total_mass = 0.0
        cm_x = 0.0
        cm_y = 0.0
        cm_z = 0.0

        for i in range(8):
            child = first_child + i
            m = tree[child]['mass']
            total_mass += m
            cm_x += tree[child]['pos_x'] * m
            cm_y += tree[child]['pos_y'] * m
            cm_z += tree[child]['pos_z'] * m

        tree[node]['mass'] = total_mass
        if total_mass > 0:
            tree[node]['pos_x'] = cm_x / total_mass
            tree[node]['pos_y'] = cm_y / total_mass
            tree[node]['pos_z'] = cm_z / total_mass

    return tree, node_counter