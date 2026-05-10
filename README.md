# N-Body Physics Simulator

A high-performance, interactive 3D N-body physics simulation engine written in Python. 

This project is designed for both accuracy and speed, utilizing JIT compilation and the Barnes-Hut algorithm to simulate celestial bodies in real-time, complete with interactive visualizations.

## Installation

This project uses `pyproject.toml` dependency management. You can install the core engine and choose which optional features you need.

Clone the repository and install:

```bash
# Install the core engine only (Numba, Numpy, Pint)
pip install .

# Install with 3D visualization tools (K3D, Plotly, Matplotlib)
pip install .[vis]

# Install with testing and documentation tools
pip install .[test,docs]

# Or install everything for local development!
pip install -e .[dev]
