# CFD Close Enough (Virtual Wind Tunnel)

A real-time 2D Computational Fluid Dynamics (CFD) simulator built using the **Lattice Boltzmann Method (LBM)**. This interactive tool allows you to visualize airflow over various aerodynamic shapes, modify airfoil parameters on the fly, and calculate performance metrics like Lift, Drag, and L/D Efficiency.

## Features

- **Real-Time LBM Solver:** Uses the D2Q9 lattice model for stable, fast fluid simulation.
- **Airfoil Generators:**
  - **NACA 4-Digit Series:** Dynamically generate airfoils by specifying or modifying camber, position, and thickness.
  - **Kline-Fogleman (KFM) Series:** Support for stepped airfoils (KFM-1 to KFM-4).
- **Interactive Controls:**
  - Draw or erase custom obstacles directly in the wind tunnel.
  - Adjust Angle of Attack (AoA) and flow velocity.
- **Live Analytics HUD:**
  - Lift and Drag Forces (N).
  - Max Velocity & Dynamic Pressure.
  - L/D Efficiency scaling.

## Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install numpy pygame
```

## How to Run

Execute the main script to start the simulation:

```bash
python main.py
```

## Controls

### Simulation Controls
- `R`: Reset simulation state.
- `C`: Clear all obstacles.
- `Left Click`: Draw custom obstacles.
- `Right Click`: Erase obstacles.
- `Up / Down Arrows`: Increase/Decrease Angle of Attack (AoA).
- `Left / Right Arrows`: Decrease/Increase Wind + Aircraft Speed.

### Airfoil Presets
- `1`: NACA 0012 (Symmetrical)
- `2`: NACA 2412 (Standard)
- `3`: NACA 4415 (High Camber)
- Type a 4-digit number and press `Enter` in the top-right box to load a custom NACA profile.

### NACA Customization
- `Q / A`: Increase / Decrease Camber.
- `W / S`: Move Camber Position Forward / Backward.
- `E / D`: Increase / Decrease Thickness.

## Project Structure

- `main.py`: Application entry point, PyGame event loop, and HUD rendering.
- `lbm_solver.py`: Core D2Q9 LBM implementation and force calculators.
- `airfoils.py`: Math for plotting NACA 4-digit and KFM profiles into grid masks.
- `visualizations.py`: Fast mapping from velocity fields to visual heatmaps.

## Testing

Run the test suite with:

```bash
pip install numpy pytest
python -m pytest -q
```

The tests in `tests/` cover the NACA/KFM airfoil mask generators and the D2Q9 LBM solver (equilibrium, stability, obstacle no-slip behaviour).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
