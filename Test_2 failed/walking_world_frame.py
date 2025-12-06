#!/usr/bin/env python3
"""
Walking controller using WORLD-FRAME trajectories (feet progress in world coordinates).
This should result in natural forward walking without the backward motion issue.
"""

import numpy as np
import mujoco
import time

# Load model and data
xml_path = "legged_robot_ik.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Load trajectories and IK solutions
base_trajectory = np.load("base_trajectory.npy")
foot1_trajectory = np.load("foot1_trajectory.npy")
foot2_trajectory = np.load("foot2_trajectory.npy")
joint_targets = np.load("joint_targets_warmstart.npy")

num_steps = base_trajectory.shape[0]

print("=" * 80)
print("WALKING CONTROLLER - WORLD-FRAME TRAJECTORIES")
print("=" * 80)
print(f"\nTrajectory points: {num_steps}")
print(f"Total forward distance: {(base_trajectory[-1, 0] - base_trajectory[0, 0])*1000:.2f}mm")

# Controller parameters
KP = 20.0
KD = 2.0
dt = model.opt.timestep

# Visualization parameters
fps = 30
slow_motion = 5
steps_per_frame = int((1.0 / fps / slow_motion) / dt)

# Tracking
hip_positions = []
foot1_pos_history = []
foot2_pos_history = []
foot1_errors = []
foot2_errors = []
joint_angles_history = []

print(f"\nController parameters:")
print(f"  KP: {KP}, KD: {KD}")
print(f"  Timestep: {dt}s")
print(f"  Display FPS: {fps} ({slow_motion}x slow-motion)")
print(f"  Steps per frame: {steps_per_frame}")

# Get foot site IDs
foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot1_site")
foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot2_site")

# Initialize
data.qpos[:] = 0
mujoco.mj_forward(model, data)

# Set initial pose to first IK solution
data.qpos[3:9] = joint_targets[0, :]
mujoco.mj_forward(model, data)

last_angles = data.qpos[3:9].copy()
frame_count = 0
sim_step_count = 0
cycle_count = 0

print("\n" + "=" * 80)
print("SIMULATING...")
print("=" * 80 + "\n")

# Run simulation
max_sim_steps = num_steps * 5  # 5 complete trajectories
while sim_step_count < max_sim_steps:
    # Use modulo to cycle through trajectories
    traj_idx = sim_step_count % num_steps
    
    # Get target joint angles
    target_angles = joint_targets[traj_idx, :].copy()
    current_angles = data.qpos[3:9].copy()
    
    # PD control
    angle_errors = target_angles - current_angles
    angle_velocities = (current_angles - last_angles) / dt
    velocity_errors = -angle_velocities
    
    control = KP * angle_errors + KD * velocity_errors
    data.ctrl[:] = np.clip(control, -1.0, 1.0)
    
    # Step physics
    mujoco.mj_step(model, data)
    
    last_angles = current_angles.copy()
    sim_step_count += 1
    
    # Record tracking data
    if sim_step_count % steps_per_frame == 0:
        hip_pos = data.xpos[model.body("hip").id].copy()
        f1_pos = data.site_xpos[foot1_id].copy()
        f2_pos = data.site_xpos[foot2_id].copy()
        
        hip_positions.append(hip_pos)
        foot1_pos_history.append(f1_pos)
        foot2_pos_history.append(f2_pos)
        
        # Track IK errors
        f1_error = np.linalg.norm(f1_pos - foot1_trajectory[traj_idx, :])
        f2_error = np.linalg.norm(f2_pos - foot2_trajectory[traj_idx, :])
        foot1_errors.append(f1_error)
        foot2_errors.append(f2_error)
        
        frame_count += 1
        
        # Print progress
        if frame_count % int(num_steps / 50) == 0:
            cycle_fraction = (sim_step_count / num_steps) % 1.0
            print(f"Step {sim_step_count:5d} ({cycle_fraction:.1%}): "
                  f"Foot1 error: {f1_error*1000:6.2f}mm, "
                  f"Foot2 error: {f2_error*1000:6.2f}mm, "
                  f"Hip X: {hip_pos[0]:8.6f}m")

hip_positions = np.array(hip_positions)
foot1_pos_history = np.array(foot1_pos_history)
foot2_pos_history = np.array(foot2_pos_history)
foot1_errors = np.array(foot1_errors) * 1000  # Convert to mm
foot2_errors = np.array(foot2_errors) * 1000

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80 + "\n")

print(f"Stride: {(base_trajectory[-1, 0] - base_trajectory[0, 0])*1000:.2f}mm per trajectory cycle")
print(f"\nTracking Performance:")
print(f"  Foot1 Mean error: {foot1_errors.mean():.2f}mm")
print(f"  Foot1 Max error:  {foot1_errors.max():.2f}mm")
print(f"  Foot2 Mean error: {foot2_errors.mean():.2f}mm")
print(f"  Foot2 Max error:  {foot2_errors.max():.2f}mm")

print(f"\nHip Forward Progression:")
print(f"  Start X: {hip_positions[0, 0]:.6f}m")
print(f"  End X:   {hip_positions[-1, 0]:.6f}m")
print(f"  Total forward motion: {(hip_positions[-1, 0] - hip_positions[0, 0])*1000:.2f}mm")
print(f"  Average per trajectory: {(hip_positions[-1, 0] - hip_positions[0, 0])/5*1000:.2f}mm")

print(f"\nZ Height variation:")
print(f"  Min: {hip_positions[:, 2].min():.6f}m")
print(f"  Max: {hip_positions[:, 2].max():.6f}m")
print(f"  Range: {(hip_positions[:, 2].max() - hip_positions[:, 2].min())*1000:.2f}mm")

# Check if walking succeeded
expected_forward = 20  # mm per trajectory
achieved_forward = (hip_positions[-1, 0] - hip_positions[0, 0]) * 1000
efficiency = achieved_forward / (expected_forward * 5) * 100  # 5 trajectories

if achieved_forward > 5:  # At least 5mm forward per trajectory
    print(f"\n✓ SUCCESS: Robot is walking forward!")
    print(f"  Expected: {expected_forward}mm per trajectory")
    print(f"  Achieved: {achieved_forward/5:.2f}mm per trajectory")
    print(f"  Efficiency: {efficiency:.1f}%")
else:
    print(f"\n✗ FAILED: Insufficient forward motion")
    print(f"  Expected: {expected_forward}mm per trajectory")
    print(f"  Achieved: {achieved_forward/5:.2f}mm per trajectory")

print("\n" + "=" * 80)
