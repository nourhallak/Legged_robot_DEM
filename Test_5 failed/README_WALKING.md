# Biped Robot Walking Trajectory Generation and Simulation

## Overview

This package generates realistic bipedal walking trajectories for a legged robot and provides tools for inverse kinematics solving and MuJoCo simulation.

## Scripts

### 1. **generate_walking_trajectories.py**
Generates smooth, physically-valid bipedal walking trajectories.

**Output files:**
- `base_trajectory.npy` - Base (hip) position trajectory (400 steps, 3 coordinates)
- `foot1_trajectory.npy` - Left foot position trajectory
- `foot2_trajectory.npy` - Right foot position trajectory
- `walking_trajectories.png` - Visualization of trajectories

**Configuration parameters:**
- `NUM_STEPS`: Total trajectory points (default: 400)
- `STRIDE_LENGTH`: Forward progress per step (default: 5 mm)
- `CYCLE_STEPS`: Steps per half gait cycle (default: 100)
- `STANCE_FRACTION`: Percent of cycle in stance phase (default: 60%)
- `GROUND_Z`: Ground contact height (default: 210 mm)
- `SWING_CLEARANCE`: Foot clearance during swing (default: 10 mm)
- `FOOT_SPACING`: Left-right foot separation (default: 20 mm)

**Usage:**
```bash
python generate_walking_trajectories.py
```

### 2. **solve_ik.py**
Solves inverse kinematics to find joint angles matching foot trajectories.

**Requirements:**
- Generated walking trajectories (base_trajectory.npy, etc.)
- Model file (legged_robot_ik.xml)

**Output files:**
- `left_leg_angles.npy` - Left leg joint angles (N, 3)
- `right_leg_angles.npy` - Right leg joint angles (N, 3)
- `left_leg_success.npy` - IK success flags for left leg
- `right_leg_success.npy` - IK success flags for right leg
- `left_leg_errors.npy` - Position errors for left leg
- `right_leg_errors.npy` - Position errors for right leg

**Configuration:**
- `kp`: Proportional gain for IK (increase for tighter convergence)
- `kd`: Derivative gain for IK
- `max_iter`: Maximum optimization iterations per point (default: 100)

**Usage:**
```bash
python solve_ik.py
```

### 3. **run_walking_sim.py**
Runs MuJoCo simulation with PD control tracking joint trajectories.

**Requirements:**
- IK-solved joint angles (from solve_ik.py)
- Model file (legged_robot_ik.xml)

**Configuration:**
- `duration`: Simulation time in seconds (default: 5.0)
- `render`: Enable visualization (default: False)
- `kp`: Proportional controller gain (default: 100)
- `kd`: Derivative controller gain (default: 10)

**Usage:**
```bash
python run_walking_sim.py
```

### 4. **analyze_trajectories.py**
Generates detailed analysis and visualization of trajectories.

**Output files:**
- `trajectory_report.txt` - Text report with statistics
- `detailed_analysis.png` - Comprehensive analysis plots

**Analysis includes:**
- Position and velocity statistics
- Gait phase information (stance vs swing)
- Foot separation distance
- Side and front views
- Joint angle ranges (if available)

**Usage:**
```bash
python analyze_trajectories.py
```

### 5. **quick_start.py**
Automated pipeline to run all steps sequentially.

**Pipeline:**
1. Generate trajectories
2. Analyze trajectories
3. Solve IK (if available)
4. Run simulation (if available)

**Usage:**
```bash
python quick_start.py
```

## Trajectory Format

All trajectory files are NumPy arrays (.npy) with shape (N, 3) where:
- N = number of trajectory points
- 3 = X, Y, Z coordinates in meters

### Coordinate System
- X: Forward direction (robot walking direction)
- Y: Lateral direction (left-right)
- Z: Vertical direction (up-down)

### Default Heights (in meters)
- Ground level: 0.210 m (210 mm)
- Swing height: 0.220 m (10 mm clearance)
- Hip height: 0.215-0.225 m (oscillates)

## Gait Parameters

The generated gait is a standard bipedal walking pattern with:
- **Stride length**: 5 mm forward per step
- **Gait cycle**: 100 steps (alternating feet)
- **Stance phase**: 60 steps (foot on ground)
- **Swing phase**: 40 steps (foot in air)
- **Foot-to-foot distance**: 20 mm (left-right)

## Validation Checks

The trajectory generator automatically validates:
1. ✓ Feet stay above ground level
2. ✓ Swing heights within limits
3. ✓ Forward motion is monotonic (no backward stepping)
4. ✓ Walking in straight line (no lateral deviation)
5. ✓ Feet alternate properly (no simultaneous flight)

## Troubleshooting

### Missing IK solutions
If `solve_ik.py` fails, check:
- Model file exists and is valid
- Foot trajectories are reachable by the robot
- Joint angle bounds are set correctly
- Try increasing `max_iter` for better convergence

### Simulation errors
If simulation crashes:
- Check that IK solutions exist
- Verify joint angles are within valid ranges
- Reduce simulation timestep in model XML
- Check PD controller gains (`kp`, `kd`)

### Poor trajectory tracking
If simulation doesn't track trajectories well:
- Increase proportional gain (`kp`)
- Check foot trajectories are physically achievable
- Verify IK solver convergence (check error files)
- Reduce stride length or increase swing clearance

## File Dependencies

```
generate_walking_trajectories.py
├── Input: legged_robot_ik.xml (model)
└── Output: base_trajectory.npy, foot1_trajectory.npy, foot2_trajectory.npy

solve_ik.py
├── Input: base_trajectory.npy, foot1_trajectory.npy, foot2_trajectory.npy
├── Input: legged_robot_ik.xml (model)
└── Output: left_leg_angles.npy, right_leg_angles.npy, *_errors.npy

run_walking_sim.py
├── Input: left_leg_angles.npy, right_leg_angles.npy
├── Input: legged_robot_ik.xml (model)
└── Output: Simulation data

analyze_trajectories.py
├── Input: base_trajectory.npy, foot1_trajectory.npy, foot2_trajectory.npy
├── Input: left_leg_angles.npy, right_leg_angles.npy (optional)
└── Output: trajectory_report.txt, detailed_analysis.png
```

## Example Workflow

```bash
# Step 1: Generate trajectories
python generate_walking_trajectories.py

# Step 2: Analyze before IK
python analyze_trajectories.py

# Step 3: Solve inverse kinematics
python solve_ik.py

# Step 4: Run simulation
python run_walking_sim.py

# Step 5: Post-analysis
python analyze_trajectories.py
```

Or simply:
```bash
python quick_start.py
```

## Performance Notes

- **Trajectory generation**: ~1 second
- **IK solving** (400 points): ~30-60 seconds
- **Simulation** (5 seconds @ 1kHz): ~5-10 seconds
- **Analysis**: ~2-5 seconds

## References

- Inverted pendulum model for hip height oscillation
- Sinusoidal foot swing trajectory
- PD joint controllers for trajectory tracking
- Numerical IK using L-BFGS-B optimization

## Author

DEM Project - Fall 2025  
MECH 620: Intermediate Dynamics
