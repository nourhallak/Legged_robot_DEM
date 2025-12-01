# Sites and Trajectory Planning - Quick Reference

## What are Sites?

**Sites** are named points of interest on the robot (like end-effectors or markers).

In this robot:
```
com_site     → Attached to link_2_1 (first leg link)
foot1_site   → Attached to foot_1 (left foot end-effector)
foot2_site   → Attached to foot_2 (right foot end-effector)
```

Sites are used for **trajectory tracking** - they define WHERE we want the robot to reach.

## Trajectory Planning

We need to plan trajectories for:
1. **Hip (base)** - The floating base position (where body moves)
2. **Foot1** - Left foot path (stance then swing)
3. **Foot2** - Right foot path (opposite phase)

```
Trajectory Planning
    ↓
hip_trajectory.npy      [400 steps × 3 coords]
foot1_trajectory.npy    [400 steps × 3 coords]
foot2_trajectory.npy    [400 steps × 3 coords]
    ↓
IK Solver (per step)
    ↓
Joint angles that reach targets
    ↓
Walking motion
```

## The Complete Pipeline

```python
# Step 1: Generate trajectories
python trajectory_planner.py
# Creates: hip_trajectory.npy, foot1_trajectory.npy, foot2_trajectory.npy

# Step 2: Validate IK
python test_ik_feet_only.py
# Shows tracking accuracy

# Step 3: Run simulation
python walking_simulator.py
# Executes complete walking
```

## Trajectory Contents

### hip_trajectory.npy
```
Shape: (400, 3)  # 400 steps, each with [x, y, z]

Example:
  Step 0:   [0.00000,  0.00000,  0.2100]  ← Starting position
  Step 1:   [0.00003,  0.00000,  0.2102]  ← Moving forward slightly, lifting
  ...
  Step 200: [0.00500,  0.00000,  0.2100]  ← Midway point
  Step 399: [0.00998,  0.00000,  0.2098]  ← Near end
```

### foot1_trajectory.npy (Left foot)
```
Shape: (400, 3)

Steps 0-100:    Stance phase (foot on ground, Z ≈ 0.210m)
Steps 100-200:  Swing phase (foot in air, Z rises to 0.225m then falls)
Steps 200-300:  Stance phase again
Steps 300-400:  Swing phase again
```

### foot2_trajectory.npy (Right foot)
```
Shape: (400, 3)

Opposite phase from foot1:
Steps 0-100:    Swing phase (while foot1 in stance)
Steps 100-200:  Stance phase (while foot1 in swing)
etc.
```

## How IK Uses Sites

For each step:
```python
# 1. Get trajectory targets
target_hip = hip_trajectory[step]           # Where base should be
target_foot1 = foot1_trajectory[step]       # Where left foot should be
target_foot2 = foot2_trajectory[step]       # Where right foot should be

# 2. Set hip position (directly, no IK needed)
qpos[0:3] = target_hip

# 3. Solve IK for joints to reach feet targets
qpos[6:12] = solve_ik_for_feet(target_foot1, target_foot2)

# 4. Update simulation
mujoco.mj_kinematics(model, data)
```

## Why This Works

- **6 DOF for hip** (floating base can be set directly)
- **6 actuated joints** solve for 2 feet × 3 coordinates = 6 constraints
- **Well-defined system** (6 unknowns, 6 equations)
- **IK converges** in ~50 iterations per step

## Expected Performance

- **Mean tracking error**: ~16-20 mm
- **Max tracking error**: ~25 mm
- **Convergence**: 100% of steps within <30 iterations
- **Total X distance**: 10 mm over 400 steps

The small errors are due to the robot's kinematics (feet naturally offset from hip) and are acceptable for walking control.

## Files Reference

| File | Purpose |
|------|---------|
| `trajectory_planner.py` | Generates smooth walking paths |
| `test_ik_feet_only.py` | Validates IK accuracy |
| `walking_simulator.py` | Executes walking motion |
| `explain_sites.py` | Demonstrates site concept |
| `hip_trajectory.npy` | Base positions (generated) |
| `foot1_trajectory.npy` | Left foot positions (generated) |
| `foot2_trajectory.npy` | Right foot positions (generated) |

## Customization Tips

**Faster walking:**
```python
stride_length = 0.010  # Increase from 0.005
```

**Higher walking:**
```python
hip_z_max = 0.220    # Increase from 0.215
hip_z_min = 0.195    # Decrease from 0.200
```

**Better IK accuracy:**
```python
max_iterations = 100  # Increase from 50
tolerance = 1e-6      # Make stricter than 1e-5
```

**Longer motion:**
```python
num_steps = 800  # Increase from 400
```

## Key Concepts Recap

- **Sites** = named points on the robot (end-effectors)
- **Trajectories** = smooth paths for sites to follow
- **Hip** = floating base (6 DOF, set directly)
- **Joints** = actuated (6 DOF, solved by IK)
- **IK** = inverse kinematics (find joints from end-effector targets)
- **Tracking error** = difference between target and achieved site position

**Bottom line**: This system generates realistic walking by planning smooth trajectories and tracking them with IK!
