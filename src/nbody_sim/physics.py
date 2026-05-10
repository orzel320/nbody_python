import numpy as np
from numba import njit

@njit(fastmath=True)
def calculate_acceleration(positions, masses, epsilon_sq=0.0):
    """
    Calculates the exact gravitational acceleration between all pairs of bodies.
    This is an O(N^2) operation, used for base testing and small clusters.
    """
    n = positions.shape[0]
    acc = np.zeros((n, 3), dtype=np.float64) # <-- Changed to 3

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            
            d_sq = dx**2 + dy**2 + dz**2
            dist = np.sqrt(d_sq)
            denom = (d_sq + epsilon_sq) * dist
            
            if denom > 0:
                acc_mag = masses[j] / denom
                acc[i, 0] += dx * acc_mag
                acc[i, 1] += dy * acc_mag
                acc[i, 2] += dz * acc_mag
                
    return acc

@njit(fastmath=True)
def calculate_barnes_hut_accel(positions, masses, tree, body_indices, theta_sq=1.0, epsilon_sq=0.01):
    """
    Calculates 3D gravitational acceleration using the O(N log N) Barnes-Hut algorithm.
    """
    n = positions.shape[0]
    acc = np.zeros((n, 3), dtype=np.float64) 
    
    for i in range(n):
        pos_x = positions[i, 0]
        pos_y = positions[i, 1]
        pos_z = positions[i, 2]
        
        stack = np.zeros(256, dtype=np.int32)
        stack_ptr = 0
        
        stack[stack_ptr] = 0
        stack_ptr += 1
        
        while stack_ptr > 0:
            stack_ptr -= 1
            node_idx = stack[stack_ptr]
            node = tree[node_idx]
            
            dx = node['pos_x'] - pos_x
            dy = node['pos_y'] - pos_y
            dz = node['pos_z'] - pos_z
            d_sq = dx**2 + dy**2 + dz**2
            
            if node['mass'] == 0.0:
                continue
                
            if node['size']**2 < d_sq * theta_sq:
                dist = np.sqrt(d_sq)
                denom = (d_sq + epsilon_sq) * dist
                if denom > 0:
                    acc_mag = node['mass'] / denom
                    acc[i, 0] += dx * acc_mag
                    acc[i, 1] += dy * acc_mag
                    acc[i, 2] += dz * acc_mag
                    
            elif node['children'] == -1:
                start = node['body_start']
                end = node['body_end']
                for idx in range(start, end):
                    j = body_indices[idx]
                    
                    if i == j:
                        continue
                        
                    dx_j = positions[j, 0] - pos_x
                    dy_j = positions[j, 1] - pos_y
                    dz_j = positions[j, 2] - pos_z
                    d_sq_j = dx_j**2 + dy_j**2 + dz_j**2
                    
                    dist_j = np.sqrt(d_sq_j)
                    denom_j = (d_sq_j + epsilon_sq) * dist_j
                    if denom_j > 0:
                        acc_mag = masses[j] / denom_j
                        acc[i, 0] += dx_j * acc_mag
                        acc[i, 1] += dy_j * acc_mag
                        acc[i, 2] += dz_j * acc_mag
                        
            else:
                child_start = node['children']
                for c in range(8):
                    stack[stack_ptr] = child_start + c
                    stack_ptr += 1
                    
    return acc

@njit(fastmath=True)
def update_bodies(positions, velocities, accelerations, dt):
    """
    Updates velocities and positions using Symplectic Euler integration.
    Operates entirely in-place to avoid memory allocation overhead.
    """
    n = positions.shape[0]
    for i in range(n):
        velocities[i, 0] += accelerations[i, 0] * dt
        velocities[i, 1] += accelerations[i, 1] * dt
        velocities[i, 2] += accelerations[i, 2] * dt 
        
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt
        positions[i, 2] += velocities[i, 2] * dt