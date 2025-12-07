# 2D Biped Robot Walking on Sand - DEM Simulation

## Overview

This project simulates a 2D biped robot walking on granular sand using the Discrete Element Method (DEM). The simulation models particle-to-particle interactions, foot-sand contact physics, and realistic walking kinematics to visualize how the robot deforms sand as it walks.

## Features

- **Robot Kinematics**: Full 3-DOF leg with inverse and forward kinematics
- **DEM Sand Physics**: Individual particle dynamics with collision detection
- **Realistic Contact**: Rectangular foot-sand contact zones with spring-damper model
- **Particle Interactions**: Particle-to-particle collisions with gravity and damping
- **Walking Gait**: Multi-step walking trajectory with configurable parameters
- **Visualization**: Real-time frame rendering and GIF export
- **Analysis Graphs**: Normal force and compaction tracking over time

## System Requirements

- Python 3.6+
- NumPy
- PIL (Pillow)
- Matplotlib

## Installation

```bash
pip install numpy pillow matplotlib
```

## Usage

Run the simulation:

```bash
python simulation_runner.py
```

## Output Files

The simulation generates the following files in the `output` directory:

- `biped_walking.gif` - Animated visualization of the robot walking
- `normal_force_plot.png` - Graph of normal forces vs time
- `compaction_plot.png` - Graph of sand compaction vs time
- `combined_plot.png` - Combined visualization of both metrics

## Configuration Parameters

Key parameters that can be adjusted in the code:

### Robot Geometry
- `L_thigh = 0.35` - Thigh length (m)
- `L_shank = 0.35` - Shank length (m)
- `L_foot = 0.1` - Foot length (m)
- `hip_height = 0.6` - Hip height above ground (m)

### Robot Mass Distribution
- `mass_torso = 10.0` - Central body mass (kg)
- `mass_thigh = 3.5` - Each thigh mass (kg)
- `mass_shank = 2.5` - Each shank mass (kg)
- `mass_foot = 1.0` - Each foot mass (kg)
- `total_mass = 24.0` - Total robot mass (kg)

### Walking
- `step_length = 0.3` - Distance per step (m)
- `step_height = 0.12` - Maximum swing height (m)
- `num_steps = 6` - Number of steps to simulate
- `frames_per_step = 60` - Animation frames per step

### Sand Properties
- `particle_radius = 0.015` - Radius of each sand particle (m)
- `grid_spacing = particle_radius * 2.1` - Initial particle spacing (tighter = more packed)

### Physics
- `stiffness = 500.0` - Spring constant for foot-sand contact (sand compression)
- `particle_stiffness = 1000.0` - Spring constant for particle-particle collisions
- `gravity = 0.5` - Gravitational acceleration (very weak to keep sand stable)
- `g = 9.81` - Standard gravity constant (used for foot weight calculation)

### Container
- `sand_container_left = -0.2` - Left boundary
- `sand_container_right = 2.0` - Right boundary
- `sand_container_bottom = -0.3` - Bottom boundary
- `sand_container_top = 0.2` - Top boundary

## How It Works

### Walking Trajectory
The robot follows a sinusoidal swing trajectory for the stepping foot while the stance foot remains planted. The stance foot switches every 60 frames, creating a continuous walking gait.

### Contact Physics (Mass-Based with Dynamic Weight Distribution)
Normal force is calculated from the robot's weight and is dynamically distributed based on the number of feet in contact:
- **Single support (1 foot)**: Full robot weight (24 kg × 9.81 m/s² = 235.4 N)
- **Double support (2 feet)**: Weight split equally (117.7 N per foot)
- **No support (airborne)**: Zero force

This weight is distributed evenly across all particles in the contact zone. The sand compression depth is calculated as δ = F / k_stiffness. Forces are applied as **velocity impulses** (not position changes) to particles, which allows natural damping and prevents oscillations.

### Particle Dynamics
Sand particles interact with each other through elastic collisions, apply weak gravity to airborne particles, and are constrained within the container boundaries. The simulation runs at 0.01 second time steps. Velocity impulses are used instead of direct position modifications to ensure physically correct behavior.

### Data Collection
The simulation tracks:
- Normal forces on each foot during contact (dynamic: 235.4 N during single support, 117.7 N during double support)
- Sand particle compaction levels
- Sand compression depth from spring deflection
- Overall packing density

## Physics Validation

- **Normal Force**: Single support ~235.4 N, double support ~117.7 N per foot, zero when airborne
- **Spring Deflection**: Calculated as δ = F/k (0.470 m during single support, 0.235 m during double support)
- **Velocity Impulses**: Applied to particle velocities (not positions) for stable dynamics
- **Dynamic Weight Distribution**: Automatically adjusts based on number of feet in contact
- **Gravity**: Only applied to airborne particles; grounded particles are supported

## Output Analysis

### Normal Force Graphs
- Shows distinct contact and swing phases
- Left and right feet have opposite phase relationships
- Force magnitude correlates with walking pace

### Compaction Graphs
- Increases as robot steps on sand
- Reflects sand compression and rearrangement
- Provides measure of energy dissipation

## Future Enhancements

- 3D simulation for more realistic physics
- Variable walking speed and gait patterns
- Multiple robots walking simultaneously
- Terrain slope and obstacles
- Energy consumption analysis
- Spatial hashing for better performance with more particles

## References

- Discrete Element Method fundamentals
- Robot kinematics and inverse kinematics
- Spring-damper collision models
- Granular mechanics physics

## Author Notes

The simulation prioritizes physical accuracy and visual clarity. All parameters are tunable for different scenarios and research purposes.
