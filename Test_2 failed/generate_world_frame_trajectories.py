#!/usr/bin/env python3
"""
Generate WORLD-FRAME walking trajectories where feet progress forward in world coordinates.
This ensures the body moves forward naturally as feet plant ahead.
"""

import numpy as np
import mujoco

xml_path = "legged_robot_ik.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("=" * 80)
print("GENERATING WORLD-FRAME WALKING TRAJECTORIES")
print("=" * 80)

# Find standing pose
print("\nFinding standing pose...")
def get_feet_positions(data, model):
    foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot1_site")
    foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot2_site")
    return data.site_xpos[foot1_id].copy(), data.site_xpos[foot2_id].copy()

data.qpos[:] = 0
mujoco.mj_forward(model, data)
f1_stand, f2_stand = get_feet_positions(data, model)

print(f"Standing foot positions:")
print(f"  Foot1: {f1_stand}")
print(f"  Foot2: {f2_stand}")

# Trajectory parameters
total_steps = 400
ground_z = (f1_stand[2] + f2_stand[2]) / 2
swing_z = ground_z + 0.006  # 6mm clearance
hip_height = 0.42

foot1_y = f1_stand[1]
foot2_y = f2_stand[1]
hip_y = (foot1_y + foot2_y) / 2

# Walking parameters - small incremental forward motion
stride_length = 0.00005  # 0.05mm per step (5mm stride per 100 steps)
num_steps_per_cycle = 100  # 100 steps per walking cycle
num_cycles = total_steps // num_steps_per_cycle

print(f"\nTrajectory parameters:")
print(f"  Total steps: {total_steps}")
print(f"  Steps per cycle: {num_steps_per_cycle}")
print(f"  Number of cycles: {num_cycles}")
print(f"  Stride per step: {stride_length*1000:.2f}mm")
print(f"  Total forward distance: {stride_length * total_steps*1000:.2f}mm")

# Initialize trajectories
base_trajectory = np.zeros((total_steps, 3))
foot1_trajectory = np.zeros((total_steps, 3))
foot2_trajectory = np.zeros((total_steps, 3))
com_trajectory = np.zeros((total_steps, 3))

# Generate trajectories
for step in range(total_steps):
    # Hip moves continuously forward
    base_trajectory[step, 0] = stride_length * step
    base_trajectory[step, 1] = hip_y
    base_trajectory[step, 2] = hip_height
    
    com_trajectory[step, :] = base_trajectory[step, :]
    
    # Determine which leg is stance vs swing
    step_in_cycle = step % num_steps_per_cycle
    stance_phase_end = int(num_steps_per_cycle * 0.68)  # 68% stance
    
    if step_in_cycle < stance_phase_end:  # Stance phase
        # Leg stays planted
        if step < num_steps_per_cycle // 2:
            # Foot1 on ground
            foot1_plant_x = stride_length * step
            foot1_trajectory[step, 0] = foot1_plant_x
            foot1_trajectory[step, 1] = foot1_y
            foot1_trajectory[step, 2] = ground_z
            
            # Foot2 swinging from previous plant
            foot2_plant_x = stride_length * (step - num_steps_per_cycle // 2)
            foot2_trajectory[step, 0] = foot2_plant_x
            foot2_trajectory[step, 1] = foot2_y
            foot2_trajectory[step, 2] = ground_z
        else:
            # Foot2 on ground
            foot2_plant_x = stride_length * (step - num_steps_per_cycle // 2)
            foot2_trajectory[step, 0] = foot2_plant_x
            foot2_trajectory[step, 1] = foot2_y
            foot2_trajectory[step, 2] = ground_z
            
            # Foot1 planted from previous contact
            foot1_plant_x = stride_length * (step - num_steps_per_cycle)
            foot1_trajectory[step, 0] = foot1_plant_x
            foot1_trajectory[step, 1] = foot1_y
            foot1_trajectory[step, 2] = ground_z
    else:  # Swing phase
        # Feet swing forward to next plant position
        swing_norm = (step_in_cycle - stance_phase_end) / (num_steps_per_cycle - stance_phase_end)
        
        if step < num_steps_per_cycle // 2:
            # Foot2 swings
            current_plant_x = stride_length * (step - num_steps_per_cycle // 2)
            next_plant_x = current_plant_x + num_steps_per_cycle // 2 * stride_length
            foot2_trajectory[step, 0] = current_plant_x + (next_plant_x - current_plant_x) * swing_norm
            foot2_trajectory[step, 1] = foot2_y
            lift = (swing_z - ground_z) * np.sin(np.pi * swing_norm)
            foot2_trajectory[step, 2] = ground_z + lift
            
            # Foot1 stays
            foot1_plant_x = stride_length * step
            foot1_trajectory[step, 0] = foot1_plant_x
            foot1_trajectory[step, 1] = foot1_y
            foot1_trajectory[step, 2] = ground_z
        else:
            # Foot1 swings
            current_plant_x = stride_length * (step - num_steps_per_cycle)
            next_plant_x = current_plant_x + num_steps_per_cycle // 2 * stride_length
            foot1_trajectory[step, 0] = current_plant_x + (next_plant_x - current_plant_x) * swing_norm
            foot1_trajectory[step, 1] = foot1_y
            lift = (swing_z - ground_z) * np.sin(np.pi * swing_norm)
            foot1_trajectory[step, 2] = ground_z + lift
            
            # Foot2 stays
            foot2_plant_x = stride_length * (step - num_steps_per_cycle // 2)
            foot2_trajectory[step, 0] = foot2_plant_x
            foot2_trajectory[step, 1] = foot2_y
            foot2_trajectory[step, 2] = ground_z

# Save trajectories
np.save("base_trajectory.npy", base_trajectory)
np.save("foot1_trajectory.npy", foot1_trajectory)
np.save("foot2_trajectory.npy", foot2_trajectory)
np.save("com_trajectory.npy", com_trajectory)

print(f"\n{'='*80}")
print("TRAJECTORY STATISTICS")
print(f"{'='*80}\n")

print(f"Hip trajectory:")
print(f"  X: {base_trajectory[0, 0]:.6f} to {base_trajectory[-1, 0]:.6f}")
print(f"  Total forward: {(base_trajectory[-1, 0] - base_trajectory[0, 0])*1000:.2f}mm")

print(f"\nFoot1 trajectory:")
print(f"  X: {foot1_trajectory[:, 0].min():.6f} to {foot1_trajectory[:, 0].max():.6f}")
print(f"  Y: {foot1_trajectory[0, 1]:.6f}")
print(f"  Z: {foot1_trajectory[:, 2].min():.6f} to {foot1_trajectory[:, 2].max():.6f}")

print(f"\nFoot2 trajectory:")
print(f"  X: {foot2_trajectory[:, 0].min():.6f} to {foot2_trajectory[:, 0].max():.6f}")
print(f"  Y: {foot2_trajectory[0, 1]:.6f}")
print(f"  Z: {foot2_trajectory[:, 2].min():.6f} to {foot2_trajectory[:, 2].max():.6f}")

print(f"\nTrajectories saved!")
