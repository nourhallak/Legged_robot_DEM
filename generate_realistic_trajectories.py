#!/usr/bin/env python3
"""
Generate REALISTIC walking trajectories based on robot's actual reachable workspace.
This ensures the IK can converge with low error.
"""

import numpy as np
import mujoco
from scipy.optimize import minimize

xml_path = "legged_robot_ik.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("=" * 80)
print("GENERATING REALISTIC TRAJECTORIES FROM ROBOT WORKSPACE")
print("=" * 80)

# First, find the actual reachable foot positions
print("\nAnalyzing robot workspace...")

def get_feet_positions(data, model):
    foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot1_site")
    foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot2_site")
    return data.site_xpos[foot1_id].copy(), data.site_xpos[foot2_id].copy()

# Sample workspace
foot_positions_foot1 = []
foot_positions_foot2 = []

for trial in range(1000):
    np.random.seed(trial)
    angles = np.random.uniform(-1.57, 1.57, 6)
    data.qpos[3:9] = angles
    mujoco.mj_forward(model, data)
    
    f1, f2 = get_feet_positions(data, model)
    foot_positions_foot1.append(f1)
    foot_positions_foot2.append(f2)

foot_positions_foot1 = np.array(foot_positions_foot1)
foot_positions_foot2 = np.array(foot_positions_foot2)

print(f"\nFoot1 workspace (from 1000 random samples):")
print(f"  X: [{foot_positions_foot1[:, 0].min():.6f}, {foot_positions_foot1[:, 0].max():.6f}]")
print(f"  Y: [{foot_positions_foot1[:, 1].min():.6f}, {foot_positions_foot1[:, 1].max():.6f}]")
print(f"  Z: [{foot_positions_foot1[:, 2].min():.6f}, {foot_positions_foot1[:, 2].max():.6f}]")

print(f"\nFoot2 workspace (from 1000 random samples):")
print(f"  X: [{foot_positions_foot2[:, 0].min():.6f}, {foot_positions_foot2[:, 0].max():.6f}]")
print(f"  Y: [{foot_positions_foot2[:, 1].min():.6f}, {foot_positions_foot2[:, 1].max():.6f}]")
print(f"  Z: [{foot_positions_foot2[:, 2].min():.6f}, {foot_positions_foot2[:, 2].max():.6f}]")

# Use typical standing pose as reference
print(f"\n{'='*80}")
print("GENERATING WALKING GAIT FROM FEASIBLE WORKSPACE")
print(f"{'='*80}\n")

# Find a good standing configuration (both feet at same height, close to ground)
def find_standing_pose():
    def objective(angles):
        data.qpos[3:9] = angles
        mujoco.mj_forward(model, data)
        f1, f2 = get_feet_positions(data, model)
        # Both feet at similar height
        z_error = (f1[2] - f2[2])**2
        # Feet as low as possible
        z_penalty = (f1[2] - foot_positions_foot1[:, 2].min())**2 + (f2[2] - foot_positions_foot2[:, 2].min())**2
        return z_error + 0.1 * z_penalty
    
    bounds = [(-1.57, 1.57)] * 6
    result = minimize(objective, np.zeros(6), method='L-BFGS-B', bounds=bounds)
    
    data.qpos[3:9] = result.x
    mujoco.mj_forward(model, data)
    return result.x

standing_angles = find_standing_pose()
data.qpos[3:9] = standing_angles
mujoco.mj_forward(model, data)
f1_stand, f2_stand = get_feet_positions(data, model)

print(f"Standing pose foot positions:")
print(f"  Foot1: {f1_stand}")
print(f"  Foot2: {f2_stand}")
print(f"  Height difference: {(f1_stand[2] - f2_stand[2])*1000:.2f}mm")

# Now generate realistic walking trajectory using this as reference
ground_z = (f1_stand[2] + f2_stand[2]) / 2  # Average foot height as "ground" contact
swing_z = ground_z + 0.006  # 6mm clearance

# Foot positions relative to hip
foot1_y = f1_stand[1]
foot2_y = f2_stand[1]
hip_height = 0.42

total_steps = 400
stride_length = 0.005  # 5mm stride (was 3.5mm)
cycles = 3  # More cycles for testing
cycle_length = total_steps // cycles

# Initialize trajectories
base_trajectory = np.zeros((total_steps, 3))
foot1_trajectory = np.zeros((total_steps, 3))
foot2_trajectory = np.zeros((total_steps, 3))
com_trajectory = np.zeros((total_steps, 3))

for step in range(total_steps):
    cycle_pos = (step % cycle_length) / cycle_length
    cycle_num = step // cycle_length
    
    # Hip trajectory
    base_trajectory[step, 0] = stride_length * cycle_num + stride_length * cycle_pos + stride_length / 2
    base_trajectory[step, 1] = (foot1_y + foot2_y) / 2
    base_trajectory[step, 2] = hip_height
    
    com_trajectory[step, :] = base_trajectory[step, :]
    
    # Foot 1
    foot1_plant_x = stride_length * cycle_num + stride_length / 2
    
    if cycle_pos < 0.68:  # Stance
        foot1_trajectory[step, 0] = foot1_plant_x
        foot1_trajectory[step, 1] = foot1_y
        foot1_trajectory[step, 2] = ground_z
    else:  # Swing
        swing_norm = (cycle_pos - 0.68) / 0.32
        next_plant_x = foot1_plant_x + stride_length
        foot1_trajectory[step, 0] = foot1_plant_x + (next_plant_x - foot1_plant_x) * swing_norm
        foot1_trajectory[step, 1] = foot1_y
        lift_height = (swing_z - ground_z) * np.sin(np.pi * swing_norm)
        foot1_trajectory[step, 2] = ground_z + lift_height
    
    # Foot 2 (180 degrees out of phase)
    cycle_pos2 = ((step + cycle_length // 2) % cycle_length) / cycle_length
    cycle_num2 = (step + cycle_length // 2) // cycle_length
    
    foot2_plant_x = stride_length * cycle_num2 + stride_length / 2
    
    if cycle_pos2 < 0.68:  # Stance
        foot2_trajectory[step, 0] = foot2_plant_x
        foot2_trajectory[step, 1] = foot2_y
        foot2_trajectory[step, 2] = ground_z
    else:  # Swing
        swing_norm = (cycle_pos2 - 0.68) / 0.32
        next_plant_x = foot2_plant_x + stride_length
        foot2_trajectory[step, 0] = foot2_plant_x + (next_plant_x - foot2_plant_x) * swing_norm
        foot2_trajectory[step, 1] = foot2_y
        lift_height = (swing_z - ground_z) * np.sin(np.pi * swing_norm)
        foot2_trajectory[step, 2] = ground_z + lift_height

# Save trajectories
np.save("base_trajectory.npy", base_trajectory)
np.save("foot1_trajectory.npy", foot1_trajectory)
np.save("foot2_trajectory.npy", foot2_trajectory)
np.save("com_trajectory.npy", com_trajectory)

print(f"\n{'='*80}")
print("TRAJECTORY STATISTICS")
print(f"{'='*80}\n")

print(f"Trajectory parameters:")
print(f"  Stride length: {stride_length}m ({stride_length*1000:.1f}mm)")
print(f"  Ground contact height: {ground_z:.6f}m")
print(f"  Swing height: {swing_z:.6f}m (clearance: {(swing_z - ground_z)*1000:.2f}mm)")
print(f"  Hip height: {hip_height}m")

print(f"\nFoot1 trajectory:")
print(f"  X range: [{foot1_trajectory[:, 0].min():.6f}, {foot1_trajectory[:, 0].max():.6f}]")
print(f"  Y: {foot1_trajectory[0, 1]:.6f}")
print(f"  Z range: [{foot1_trajectory[:, 2].min():.6f}, {foot1_trajectory[:, 2].max():.6f}]")

print(f"\nFoot2 trajectory:")
print(f"  X range: [{foot2_trajectory[:, 0].min():.6f}, {foot2_trajectory[:, 0].max():.6f}]")
print(f"  Y: {foot2_trajectory[0, 1]:.6f}")
print(f"  Z range: [{foot2_trajectory[:, 2].min():.6f}, {foot2_trajectory[:, 2].max():.6f}]")

print(f"\nHip trajectory:")
print(f"  X range: [{base_trajectory[:, 0].min():.6f}, {base_trajectory[:, 0].max():.6f}]")
print(f"  Y: {base_trajectory[0, 1]:.6f}")
print(f"  Z: {base_trajectory[0, 2]:.6f}")

print(f"\nStance/Swing analysis:")
foot1_stance = np.abs(foot1_trajectory[:, 2] - ground_z) < 0.0001
foot2_stance = np.abs(foot2_trajectory[:, 2] - ground_z) < 0.0001
print(f"  Foot1 stance: {np.sum(foot1_stance)} steps ({100*np.sum(foot1_stance)/total_steps:.1f}%)")
print(f"  Foot2 stance: {np.sum(foot2_stance)} steps ({100*np.sum(foot2_stance)/total_steps:.1f}%)")

print(f"\nTrajectories saved!")
print(f"  base_trajectory.npy")
print(f"  foot1_trajectory.npy")
print(f"  foot2_trajectory.npy")
print(f"  com_trajectory.npy")
