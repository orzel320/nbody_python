import numpy as np
import pint
import math
import json

ureg = pint.UnitRegistry()
ureg.define(f"nbody_time = 1 * year / {2 * math.pi}")
ureg.define("solar_mass = 1.98847e30 * kilogram")
ureg.define("earth_mass = 5.9722e24 * kilogram")

class Simulation:
    def __init__(self):
        """Initialises an empty N-body simulation."""
        self._positions = np.empty((0, 3), dtype=np.float64)
        self._velocities = np.empty((0, 3), dtype=np.float64)
        self._masses = np.empty(0, dtype=np.float64)

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

        if raw_positions.ndim == 2 and raw_positions.shape[1] == 2:
            raw_positions = np.column_stack((raw_positions, np.zeros(raw_positions.shape[0])))
            
        if raw_velocities.ndim == 2 and raw_velocities.shape[1] == 2:
            raw_velocities = np.column_stack((raw_velocities, np.zeros(raw_velocities.shape[0])))

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
        from .physics import calculate_barnes_hut_accel, update_bodies
        from .quadtree import build_tree
        
        if not isinstance(dt, pint.Quantity):
            dt = dt * ureg.nbody_time
        dt_raw = dt.to(ureg.nbody_time).magnitude

        n = self._positions.shape[0]
        if n == 0:
            return

        body_indices = np.arange(n, dtype=np.int32)
        tree, _ = build_tree(self._positions, self._masses, body_indices, leaf_capacity=1)

        acc = calculate_barnes_hut_accel(
            self._positions, self._masses, tree, body_indices, 
            theta_sq=1.0, epsilon_sq=0.01
        )
        
        update_bodies(self._positions, self._velocities, acc, dt_raw)
    
    @classmethod
    def from_json(cls, filepath):
        """
        Parses a JSON scenario file to generate a Simulation instance 
        alongside its rendering metadata.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        sim = cls()
        dt = data.get("dt", 0.01)
        
        sizes = []
        colours = []
        
        np.random.seed(42)
        
        for item in data["bodies"]:
            obj_type = item.get("type", "single")
            
            if obj_type == "single":
                mass = [item["mass"]] * ureg(item["mass_unit"])
                pos = [item["position"]] * ureg(item["pos_unit"])
                vel = [item["velocity"]] * ureg(item["vel_unit"])
                
                sim.add_bodies(pos, vel, mass)
                sizes.append(item.get("size", 2.0))
                colours.append(item.get("colour", [1.0, 1.0, 1.0, 1.0]))
                
            elif obj_type == "ring":
                count = item["count"]
                
                radii = np.random.normal(
                    loc=item["radius_mean"], 
                    scale=item.get("radius_std", 0.0), 
                    size=count
                )
                if "radius_min" in item and "radius_max" in item:
                    radii = np.clip(radii, item["radius_min"], item["radius_max"])
                    
                angles = np.random.uniform(0, 2 * math.pi, count)
                
                pos_x = radii * np.cos(angles)
                pos_y = radii * np.sin(angles)
                z_spread = item.get("pos_z_std", 0.0)
                pos_z = np.random.normal(0.0, z_spread, count) 
                pos = np.column_stack((pos_x, pos_y, pos_z)) * ureg("astronomical_unit")
                
                central_mass = item.get("central_mass", 1.0)
                base_speeds = np.sqrt(central_mass / radii)
                
                speed_var = np.random.normal(1.0, item.get("speed_variance_std", 0.0), count)
                speeds = base_speeds * speed_var
                
                vel_x = -speeds * np.sin(angles)
                vel_y = speeds * np.cos(angles)
                vz_spread = item.get("vel_z_std", 0.0)
                vel_z = np.random.normal(0.0, vz_spread, count) 
                vel = np.column_stack((vel_x, vel_y, vel_z)) * ureg("astronomical_unit / nbody_time")
                
                mass_mean = item.get("mass_mean", 1e-5)
                mass_std = item.get("mass_std", 0.0)
                raw_masses = np.random.normal(loc=mass_mean, scale=mass_std, size=count)
                raw_masses = np.maximum(raw_masses, 1e-10)
                mass = raw_masses * ureg(item.get("mass_unit", "earth_mass"))
                
                sim.add_bodies(pos, vel, mass)
                
                sizes.extend([item.get("size", 2.0)] * count)
                colours.extend([item.get("colour", [0.7, 0.7, 0.7, 0.4])] * count)
            
        sizes = np.array(sizes, dtype=np.float32)
        colours = np.array(colours, dtype=np.float32)
        name = data.get("name", "Simulation")
        
        return sim, {
            "dt": dt,
            "sizes": sizes,
            "colours": colours,
            "name": name
        }