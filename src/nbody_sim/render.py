import sys
import numpy as np
from vispy import app, scene
from nbody_sim.api import Simulation, ureg

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import plotly.graph_objects as go

import k3d

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python viewer.py <path_to_json>")
        sys.exit(1)
        
    filepath = sys.argv[1]
    
    sim, config = Simulation.from_json(filepath)
    
    dt = config["dt"]
    scaled_sizes = config["sizes"] * 0.02
    colours = config["colours"]
    name = config["name"]
    
    canvas = scene.SceneCanvas(keys='interactive', show=True, title=name, bgcolor='black', size=(900, 900))
    view = canvas.central_widget.add_view()
    
    view.camera = scene.TurntableCamera(elevation=30, azimuth=45, distance=15.0)
    markers = scene.visuals.Markers(parent=view.scene, spherical=True, scaling=True)
    
    markers.light_position = (10, 10, 10)
    markers.light_ambient = 0.3 
    markers.light_color = 'white'
    
    def update(ev):
        sim.step(dt * ureg.nbody_time)
        markers.set_data(
            pos=sim._positions.astype(np.float32), 
            face_color=colours, 
            size=scaled_sizes, 
            edge_width=0,
            edge_color='transparent' 
        )

    timer = app.Timer(interval='auto', connect=update, start=True)
    print(f"Loaded scenario: {name}. Running...")
    app.run()

def create_perspective_animation(sim, dt, sizes, colours, steps=300):
    """
    Steps the simulation and renders a true 3D scene using Matplotlib.
    Allows interactive orbiting and zooming via ipympl.
    """
    fig = plt.figure(figsize=(8, 8), facecolor='black', dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_facecolor('black')
    
    ax.axis('off') 
    
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    
    ax.view_init(elev=30, azim=45)

    x = sim._positions[:, 0]
    y = sim._positions[:, 1]
    z = sim._positions[:, 2]
    
    scatter = ax.scatter(
        x, y, z, 
        c=colours, 
        s=sizes * 10.0,
        edgecolors='none',
        depthshade=True 
    )

    def update(frame):
        sim.step(dt * ureg.nbody_time)
        
        scatter._offsets3d = (
            sim._positions[:, 0], 
            sim._positions[:, 1], 
            sim._positions[:, 2]
        )
        
        return scatter,

    anim = FuncAnimation(fig, update, frames=steps, interval=33, blit=False)
    
    return anim

def create_plotly_animation(sim, dt, sizes, colours, steps=200):
    """
    Pre-calculates simulation steps and bundles them into an interactive 
    Plotly WebGL 3D figure with true perspective.
    """
    plotly_colours = [
        f"rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, {c[3]})" 
        for c in colours
    ]
    
    x = sim._positions[:, 0]
    y = sim._positions[:, 1]
    z = sim._positions[:, 2]
    
    trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=sizes * 2.0,
            color=plotly_colours,
            line=dict(width=0),
        ),
        name="Celestial Bodies"
    )
    
    frames = []
    for i in range(steps):
        sim.step(dt * ureg.nbody_time)
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=sim._positions[:, 0],
                y=sim._positions[:, 1],
                z=sim._positions[:, 2]
            )],
            name=f"frame_{i}"
        ))
        
    layout = go.Layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-8, 8]),
            yaxis=dict(visible=False, range=[-8, 8]),
            zaxis=dict(visible=False, range=[-4, 4]),
            bgcolor='black',
            aspectmode='data'
        ),
        paper_bgcolor='black',
        margin=dict(l=0, r=0, b=0, t=0),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            x=0.05, y=0.95,
            buttons=[
                dict(
                    label='▶ Play Simulation',
                    method='animate',
                    args=[None, dict(frame=dict(duration=40, redraw=True), 
                                     transition=dict(duration=0),
                                     fromcurrent=True,
                                     mode='immediate')]
                ),
                dict(
                    label='⏸ Pause',
                    method='animate',
                    args=[[None], dict(frame=dict(duration=0, redraw=True), 
                                       mode='immediate', 
                                       transition=dict(duration=0))]
                )
            ]
        )]
    )
    
    return go.Figure(data=[trace], layout=layout, frames=frames)

def create_k3d_animation(sim, dt, sizes, colours, steps=200):
    """
    Pre-calculates simulation steps and bundles them into a high-performance 
    K3D WebGL figure with true 3D geometric scaling.
    """
    r = np.clip(colours[:, 0] * 255, 0, 255).astype(np.uint32)
    g = np.clip(colours[:, 1] * 255, 0, 255).astype(np.uint32)
    b = np.clip(colours[:, 2] * 255, 0, 255).astype(np.uint32)
    hex_colours = (r << 16) | (g << 8) | b

    positions_anim = {}
    
    positions_anim[0.0] = sim._positions.astype(np.float32).copy()

    for i in range(1, steps):
        sim.step(dt * ureg.nbody_time)
        
        positions_anim[float(i)] = sim._positions.astype(np.float32).copy()

    plot = k3d.plot(
        background_color=0x000000, 
        grid_visible=False, 
        axes_helper=False,
        camera_auto_fit=False
    )

    points = k3d.points(
        positions=positions_anim,
        colors=hex_colours,
        point_sizes=sizes * 0.05,
        shader='3d',
    )

    plot += points

    return plot