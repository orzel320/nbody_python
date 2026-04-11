import numpy as np
from numba import njit

@njit(fastmath=True)
def calculate_acceleration(positions, masses, epsilon_sq=0.0):
    """
    Calculates the exact gravitational acceleration between all pairs of bodies.
    This is an O(N^2) operation, used for base testing and small clusters.
    """
    n = positions.shape[0]
    acc = np.zeros((n, 2), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            
            d_sq = dx**2 + dy**2
            dist = np.sqrt(d_sq)
            denom = (d_sq + epsilon_sq) * dist
            
            if denom > 0:
                acc_mag = masses[j] / denom
                acc[i, 0] += dx * acc_mag
                acc[i, 1] += dy * acc_mag
                
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
        
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt