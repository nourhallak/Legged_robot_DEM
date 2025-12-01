#!/usr/bin/env python3
"""
Proper integrated walking with forward velocity and IK tracking.
Maintains world-frame trajectories while enforcing forward motion.
"""

import numpy as np
import mujoco

xml_path = "legged_robot_ik.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Load trajectories (world-frame) and IK solutions
base_trajectory = np.load("base_trajectory.npy")
foot1_trajectory = np.load("foot1_trajectory.npy")
foot2_trajectory = np.load("foot2_trajectory.npy")
joint_targets = np.load("joint_targets_warmstart.npy")

num_steps = base_trajectory.shape[0]
expected_stride = base_trajectory[-1, 0] - base_trajectory[0, 0]

print("=" * 80)
print("INTEGRATED BIPEDAL WALKING CONTROLLER")
print("=" * 80)
print(f"\nTrajectory parameters:")
print(f"  Total steps: {num_steps}")
print(f"  Expected stride: {expected_stride*1000:.2f}mm")

# Derive forward velocity from trajectory
# The trajectory covers expected_stride over num_steps
dt = model.opt.timestep
forward_velocity = expected_stride / (num_steps * dt) * 0.3  # 70% slower motion (ultra-slow)

KP_legs = 20.0
KD_legs = 2.0

print(f"\nController parameters:")
print(f"  Leg PD: KP={KP_legs}, KD={KD_legs}")
print(f"  Forward velocity: {forward_velocity*1000:.2f}mm/s")
print(f"  Timestep: {dt}s")

# Setup tracking
hip_positions = []
foot1_errors = []
foot2_errors = []

foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot1_site")
foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot2_site")

# Initialize
data.qpos[:] = 0
mujoco.mj_forward(model, data)
data.qpos[3:9] = joint_targets[0, :]
mujoco.mj_forward(model, data)

last_angles = data.qpos[3:9].copy()
root_x_joint_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_x")
vel_addr = model.jnt_dofadr[root_x_joint_idx]

print("\n" + "=" * 80)
print("SIMULATING...")
print("=" * 80 + "\n")

max_sim_steps = num_steps * 5
sim_step_count = 0

while sim_step_count < max_sim_steps:
    # Cycle through trajectory repeatingly
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
    
    # Force forward velocity on root_x (before and after step)
    data.qvel[vel_addr] = forward_velocity
    mujoco.mj_step(model, data)
    data.qvel[vel_addr] = forward_velocity
    
    last_angles = current_angles.copy()
    sim_step_count += 1
    
    # Record data every 5 steps
    if sim_step_count % 5 == 0:
        hip_pos = data.xpos[model.body("hip").id].copy()
        f1_pos = data.site_xpos[foot1_id].copy()
        f2_pos = data.site_xpos[foot2_id].copy()
        
        hip_positions.append(hip_pos)
        
        # Adjust trajectory targets by current cycle progress
        # The foot positions in trajectory are relative to first step
        # As robot moves forward, we need to account for that offset
        cycle_num = sim_step_count // num_steps
        trajectory_offset_x = cycle_num * expected_stride
        
        f1_target = foot1_trajectory[traj_idx, :].copy()
        f1_target[0] += trajectory_offset_x
        f2_target = foot2_trajectory[traj_idx, :].copy()
        f2_target[0] += trajectory_offset_x
        
        f1_error = np.linalg.norm(f1_pos - f1_target)
        f2_error = np.linalg.norm(f2_pos - f2_target)
        foot1_errors.append(f1_error * 1000)
        foot2_errors.append(f2_error * 1000)
        
        # Print progress
        if sim_step_count % (num_steps // 4) == 0:
            cycle_frac = (sim_step_count / num_steps) % 1.0
            print(f"Step {sim_step_count:5d}: Foot1 error {f1_error*1000:6.2f}mm, "
                  f"Foot2 error {f2_error*1000:6.2f}mm, Hip X: {hip_pos[0]:8.6f}m")

hip_positions = np.array(hip_positions)
foot1_errors = np.array(foot1_errors)
foot2_errors = np.array(foot2_errors)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80 + "\n")

print(f"Tracking Performance:")
print(f"  Foot1 Mean error: {foot1_errors.mean():.2f}mm")
print(f"  Foot1 Max error:  {foot1_errors.max():.2f}mm")
print(f"  Foot2 Mean error: {foot2_errors.mean():.2f}mm")
print(f"  Foot2 Max error:  {foot2_errors.max():.2f}mm")

print(f"\nForward Progression:")
print(f"  Start X: {hip_positions[0, 0]:.6f}m")
print(f"  End X:   {hip_positions[-1, 0]:.6f}m")
achieved_forward = (hip_positions[-1, 0] - hip_positions[0, 0]) * 1000
expected_total = expected_stride * 1000 * 5
print(f"  Total forward: {achieved_forward:.2f}mm")
print(f"  Expected: ~{expected_total:.2f}mm")
print(f"  Efficiency: {achieved_forward / expected_total * 100:.1f}%")

print(f"\nZ Height variation:")
z_min = hip_positions[:, 2].min()
z_max = hip_positions[:, 2].max()
print(f"  Min: {z_min:.6f}m")
print(f"  Max: {z_max:.6f}m")
print(f"  Range: {(z_max - z_min)*1000:.2f}mm")

if foot1_errors.mean() < 20 and achieved_forward > expected_total * 0.8:
    print(f"\n✓ SUCCESS: Walking achieved!")
else:
    print(f"\n⚠ Partial success - needs tuning")

print("\n" + "=" * 80)
