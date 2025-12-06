#!/usr/bin/env python3
"""
Visualize bipedal walking in MuJoCo viewer - SLOW and REPEATING.
Shows the robot walking in slow motion, repeating indefinitely.
"""

import numpy as np
import mujoco
import mujoco.viewer

xml_path = "legged_robot_ik.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Load trajectories and IK solutions
base_trajectory = np.load("base_trajectory.npy")
foot1_trajectory = np.load("foot1_trajectory.npy")
foot2_trajectory = np.load("foot2_trajectory.npy")
joint_targets = np.load("joint_targets_warmstart.npy")

num_steps = base_trajectory.shape[0]
expected_stride = base_trajectory[-1, 0] - base_trajectory[0, 0]

# Derive forward velocity from trajectory
dt = model.opt.timestep
forward_velocity = expected_stride / (num_steps * dt)

# SLOW DOWN: reduce forward velocity
slow_motion_factor = 5  # 5x slower
forward_velocity = forward_velocity / slow_motion_factor

KP_legs = 20.0
KD_legs = 2.0

# Setup initial pose
data.qpos[:] = 0
mujoco.mj_forward(model, data)
data.qpos[3:9] = joint_targets[0, :]
mujoco.mj_forward(model, data)

last_angles = data.qpos[3:9].copy()
root_x_joint_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_x")
vel_addr = model.jnt_dofadr[root_x_joint_idx]

foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot1_site")
foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot2_site")

print("=" * 80)
print("BIPEDAL WALKING VISUALIZATION - SLOW & REPEATING")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Stride: {expected_stride*1000:.2f}mm per cycle")
print(f"  Forward velocity: {forward_velocity*1000:.2f}mm/s (slowed {slow_motion_factor}x)")
print(f"  Repeating continuously...")
print(f"\nViewer controls:")
print(f"  Right-click + drag: Rotate view")
print(f"  Scroll: Zoom")
print(f"  Space: Pause/Resume")
print(f"  '[' / ']': Slow down / Speed up")
print(f"  Close window to exit")
print("=" * 80 + "\n")

cycle_count = 0
total_forward = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    sim_step_count = 0
    
    while True:  # Run indefinitely
        # Get trajectory index (cycles through)
        traj_idx = sim_step_count % num_steps
        
        # Every full cycle, print progress
        if traj_idx == 0 and sim_step_count > 0:
            cycle_count += 1
            hip_x = data.xpos[model.body("hip").id][0]
            total_forward = hip_x
            print(f"Cycle {cycle_count}: Hip X = {hip_x:.4f}m ({hip_x*1000:.2f}mm total forward)")
        
        # Get target joint angles
        target_angles = joint_targets[traj_idx, :].copy()
        current_angles = data.qpos[3:9].copy()
        
        # PD control for legs
        angle_errors = target_angles - current_angles
        angle_velocities = (current_angles - last_angles) / dt
        velocity_errors = -angle_velocities
        
        control_legs = KP_legs * angle_errors + KD_legs * velocity_errors
        data.ctrl[:] = np.clip(control_legs, -1.0, 1.0)
        
        # Apply forward velocity (slowed down)
        data.qvel[vel_addr] = forward_velocity
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Re-apply velocity after step
        data.qvel[vel_addr] = forward_velocity
        
        # Update viewer
        viewer.sync()
        
        last_angles = current_angles.copy()
        sim_step_count += 1
