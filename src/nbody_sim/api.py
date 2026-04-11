import numpy as np
import pint
import math

ureg = pint.UnitRegistry()
ureg.define(f"nbody_time = 1 * year / {2 * math.pi}")
ureg.define("solar_mass = 1.98847e30 * kilogram")
ureg.define("earth_mass = 5.9722e24 * kilogram")

class Simulation:
    def __init__(self):
        """Initialize the simulation with empty, raw NumPy arrays."""
        self._positions = np.empty((0, 2), dtype=np.float64)
        self._velocities = np.empty((0, 2), dtype=np.float64)
        self._masses = np.empty((0,), dtype=np.float64)

    def add_bodies(self, positions, velocities, masses):
        """
        Add bodies to the simulation. Accepts Pint quantities, 
        converts them to base SI units, and strips them for Numba.
        """
        if not isinstance(positions, pint.Quantity):
            positions = positions * ureg.meter
        if not isinstance(velocities, pint.Quantity):
            velocities = velocities * (ureg.meter / ureg.second)
        if not isinstance(masses, pint.Quantity):
            masses = masses * ureg.kilogram

        positions = positions.to(ureg.astronomical_unit)
        velocities = velocities.to(ureg.astronomical_unit / ureg.nbody_time)
        masses = masses.to(ureg.solar_mass)

        raw_positions = np.array(positions.magnitude, dtype=np.float64)
        raw_velocities = np.array(velocities.magnitude, dtype=np.float64)
        raw_masses = np.array(masses.magnitude, dtype=np.float64)

        if self._positions.size == 0:
            self._positions = raw_positions
            self._velocities = raw_velocities
            self._masses = raw_masses
        else:
            self._positions = np.vstack((self._positions, raw_positions))
            self._velocities = np.vstack((self._velocities, raw_velocities))
            self._masses = np.concatenate((self._masses, raw_masses))
            
    def step(self, dt):
        """Advances the simulation by one time step."""
        from .physics import calculate_acceleration, update_bodies
        
        # Strip the units from dt just like we did with bodies
        if not isinstance(dt, pint.Quantity):
            dt = dt * ureg.nbody_time
        dt_raw = dt.to(ureg.nbody_time).magnitude

        # 1. Calculate accelerations (O(N^2) for now, until we hook up the Quadtree)
        # We use a small epsilon to prevent infinite forces on overlap
        acc = calculate_acceleration(self._positions, self._masses, epsilon_sq=0.01)
        
        # 2. Update positions and velocities in-place
        update_bodies(self._positions, self._velocities, acc, dt_raw)