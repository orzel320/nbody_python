import json
import sys
import numpy as np
from vispy import app, scene
from nbody_sim.api import Simulation, ureg

def load_scenario(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    sim = Simulation()
    dt = data.get("dt", 0.01)
    
    sizes = []
    colours = []
    
    # Ensure random generation is reproducible based on the scenario
    np.random.seed(42)
    
    for item in data["bodies"]:
        obj_type = item.get("type", "single")
        
        if obj_type == "single":
            # FIX: Wrap single values in a list so NumPy sees them as 1D arrays
            mass = [item["mass"]] * ureg(item["mass_unit"])
            pos = [item["position"]] * ureg(item["pos_unit"])
            vel = [item["velocity"]] * ureg(item["vel_unit"])
            
            sim.add_bodies(pos, vel, mass)
            sizes.append(item.get("size", 2.0))
            colours.append(item.get("colour", [1.0, 1.0, 1.0, 1.0]))
            
        elif obj_type == "ring":
            # PROCEDURAL GENERATOR
            count = item["count"]
            
            # 1. Normal Distribution for Radii
            radii = np.random.normal(
                loc=item["radius_mean"], 
                scale=item.get("radius_std", 0.0), 
                size=count
            )
            # Clip outliers if specified
            if "radius_min" in item and "radius_max" in item:
                radii = np.clip(radii, item["radius_min"], item["radius_max"])
                
            angles = np.random.uniform(0, 2 * np.pi, count)
            pos_x = radii * np.cos(angles)
            pos_y = radii * np.sin(angles)
            z_spread = item.get("pos_z_std", 0.0)
            pos_z = np.random.normal(0.0, z_spread, count)
            pos = np.column_stack((pos_x, pos_y, pos_z)) * ureg("astronomical_unit")
            
            # 2. Base orbital speed + Normal Distribution for variance
            central_mass = item.get("central_mass", 1.0)
            base_speeds = np.sqrt(central_mass / radii)
            
            speed_var = np.random.normal(1.0, item.get("speed_variance_std", 0.0), count)
            speeds = base_speeds * speed_var
            
            vel_x = -speeds * np.sin(angles)
            vel_y = speeds * np.cos(angles)
            vz_spread = item.get("vel_z_std", 0.0)
            vel_z = np.random.normal(0.0, vz_spread, count)
            vel = np.column_stack((vel_x, vel_y, vel_z)) * ureg("astronomical_unit / nbody_time")
            
            # 3. Normal Distribution for Mass
            mass_mean = item.get("mass_mean", 1e-5)
            mass_std = item.get("mass_std", 0.0)
            
            raw_masses = np.random.normal(loc=mass_mean, scale=mass_std, size=count)
            # Clip to prevent negative or zero-mass bodies which would break gravity
            raw_masses = np.maximum(raw_masses, 1e-10) 
            
            mass = raw_masses * ureg(item.get("mass_unit", "earth_mass"))
            
            sim.add_bodies(pos, vel, mass)
            
            # Add visual properties for every generated body
            sizes.extend([item.get("size", 2.0)] * count)
            colours.extend([item.get("colour", [0.7, 0.7, 0.7, 0.4])] * count)
        
    sizes = np.array(sizes, dtype=np.float32)
    colours = np.array(colours, dtype=np.float32)
    
    return sim, dt, sizes, colours, data.get("name", "Simulation")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python viewer.py <path_to_json>")
        sys.exit(1)
        
    filepath = sys.argv[1]
    sim, dt, sizes, colours, name = load_scenario(filepath)
    
    canvas = scene.SceneCanvas(keys='interactive', show=True, title=name, bgcolor='black', size=(900, 900))
    view = canvas.central_widget.add_view()
    
    view.camera = scene.TurntableCamera(elevation=30, azimuth=45, distance=15.0)
    
    markers = scene.visuals.Markers(parent=view.scene, spherical=True, scaling=True)
    
    SCENE_SIZE_FACTOR = 0.02
    scaled_sizes = sizes * SCENE_SIZE_FACTOR
    
    def update(ev):
        sim.step(dt * ureg.nbody_time)
        markers.set_data(
            pos=sim._positions.astype(np.float32), 
            face_color=colours, 
            size=scaled_sizes, 
            edge_width=0,
            edge_color='transparent' 
        )

    # Set up lighting so the spheres have a 3D shadow (light coming from the top-right)
    markers.light_position = (10, 10, 10)
    markers.light_ambient = 0.3
    markers.light_color = 'white'

    timer = app.Timer(interval='auto', connect=update, start=True)
    print(f"Loaded scenario: {name}. Running...")
    app.run()