import time
import numpy as np
import matplotlib.pyplot as plt
from nbody_sim.api import Simulation, ureg

def run_benchmark(num_bodies, steps=100):
    """Runs a pure-math benchmark for a given number of bodies and returns time per step."""
    print(f"--- Benchmarking {num_bodies:,} bodies ---")
    
    sim = Simulation()
    np.random.seed(42)
    
    positions = np.random.uniform(-10.0, 10.0, (num_bodies, 3)) * ureg.astronomical_unit
    velocities = np.random.uniform(-1.0, 1.0, (num_bodies, 3)) * (ureg.astronomical_unit / ureg.nbody_time)
    masses = np.random.uniform(0.1, 10.0, num_bodies) * ureg.earth_mass
    
    sim.add_bodies(positions, velocities, masses)
    
    dt = 0.01 * ureg.nbody_time
    
    print("  Warming up Numba compiler...")
    sim.step(dt)
    
    print(f"  Running {steps} steps...")
    start_time = time.perf_counter()
    
    for _ in range(steps):
        sim.step(dt)
        
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    steps_per_sec = steps / total_time
    ms_per_step = (total_time / steps) * 1000
    
    print(f"  Result: {steps_per_sec:.2f} Steps/Second ({ms_per_step:.2f} ms per step)\n")
    return ms_per_step

if __name__ == '__main__':
    body_counts = [100, 1000, 5000, 10000, 100000]
    times_ms = []
    
    for count in body_counts:
        if count <= 1000:
            test_steps = 200
        elif count <= 10000:
            test_steps = 50
        else:
            test_steps = 10
            
        ms = run_benchmark(count, steps=test_steps)
        times_ms.append(ms)

    N = np.array(body_counts)
    T = np.array(times_ms)
    
    anchor_idx = 1 
    N_anchor = N[anchor_idx]
    T_anchor = T[anchor_idx]
    
    c_n2 = T_anchor / (N_anchor**2)
    T_n2 = c_n2 * (N**2)
    
    c_nlogn = T_anchor / (N_anchor * np.log(N_anchor))
    T_nlogn = c_nlogn * (N * np.log(N))
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    
    ax.plot(N, T, color='cyan', marker='o', linestyle='-', linewidth=2, markersize=8, label='Actual Engine Performance')
    ax.plot(N, T_nlogn, color='lime', linestyle='--', alpha=0.8, label='Theoretical O(N log N)')
    ax.plot(N, T_n2, color='red', linestyle='--', alpha=0.8, label='Theoretical O(N²)')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_title("N-Body Engine Performance Scaling", fontsize=14, pad=15)
    ax.set_xlabel("Number of Bodies (N)", fontsize=12)
    ax.set_ylabel("Compute Time per Step (ms)", fontsize=12)
    
    ax.grid(True, which="both", ls="--", alpha=0.2)
    ax.legend(loc='upper left', fontsize=11)
    
    plt.tight_layout()
    plt.show()