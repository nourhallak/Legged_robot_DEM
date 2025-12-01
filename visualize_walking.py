#!/usr/bin/env python3
"""
Visualize bipedal walking in MuJoCo viewer.
Shows the robot walking with proper forward motion and leg control.
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

# Visualization parameters
total_cycles = 3
max_sim_steps = num_steps * total_cycles
sim_step_count = 0

print("=" * 80)
print("BIPEDAL WALKING VISUALIZATION")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Total cycles: {total_cycles}")
print(f"  Expected stride: {expected_stride*1000:.2f}mm per cycle")
print(f"  Forward velocity: {forward_velocity*1000:.2f}mm/s")
print(f"  Duration: ~{(max_sim_steps * dt):.1f}s")
print(f"\nViewer controls:")
print(f"  Right-click + drag: Rotate view")
print(f"  Scroll: Zoom")
print(f"  Space: Pause/Resume")
print(f"  '[' / ']': Slow down / Speed up")
print("=" * 80 + "\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while sim_step_count < max_sim_steps:
        # Get trajectory index
        traj_idx = sim_step_count % num_steps
        
        # Get target joint angles
        target_angles = joint_targets[traj_idx, :].copy()
        current_angles = data.qpos[3:9].copy()
        
        # PD control for legs
        angle_errors = target_angles - current_angles
        angle_velocities = (current_angles - last_angles) / dt
        velocity_errors = -angle_velocities
        
        control_legs = KP_legs * angle_errors + KD_legs * velocity_errors
        data.ctrl[:] = np.clip(control_legs, -1.0, 1.0)
        
        # Apply forward velocity
        data.qvel[vel_addr] = forward_velocity
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Re-apply velocity after step
        data.qvel[vel_addr] = forward_velocity
        
        # Update viewer
        viewer.sync()
        
        last_angles = current_angles.copy()
        sim_step_count += 1
        
        # Print progress
        if sim_step_count % (num_steps // 2) == 0:
            cycle_num = sim_step_count // num_steps
            hip_x = data.xpos[model.body("hip").id][0]
            f1_pos = data.site_xpos[foot1_id]
            f2_pos = data.site_xpos[foot2_id]
            print(f"Cycle {cycle_num + 1}/{total_cycles}: Hip X = {hip_x:.4f}m, "
                  f"Foot1 Z = {f1_pos[2]:.4f}m, Foot2 Z = {f2_pos[2]:.4f}m")

print("\n" + "=" * 80)
print("Visualization complete!")
print("=" * 80)

# Print final statistics
hip_final = data.xpos[model.body("hip").id][0]
print(f"\nFinal hip position: {hip_final:.4f}m")
print(f"Total forward motion: {hip_final*1000:.2f}mm")
print(f"Expected motion: {expected_stride*1000*total_cycles:.2f}mm")
print(f"Efficiency: {(hip_final / (expected_stride * total_cycles))*100:.1f}%")
