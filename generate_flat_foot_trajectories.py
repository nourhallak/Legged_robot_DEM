#!/usr/bin/env python3
"""Generate simple walking trajectories with flat foot contact."""

import numpy as np

# Parameters
total_steps = 400
ground_z = 0.3495    # Ground contact height
swing_z = 0.355      # 5mm lift during swing
hip_height = 0.34    # Hip position (raised)

# Foot dimensions
foot_length = 0.008
foot_half_length = foot_length / 2

# Leg Y positions from corrected URDF
foot1_y = -0.008465  # Left leg
foot2_y = -0.001465  # Right leg

print("Generating simple flat-foot walking trajectories...")

# Initialize trajectories
base_trajectory = np.zeros((total_steps, 3))
foot1_trajectory = np.zeros((total_steps, 3))
foot2_trajectory = np.zeros((total_steps, 3))
com_trajectory = np.zeros((total_steps, 3))

stride_length = 0.02
cycles = 7  
cycle_length = total_steps // cycles

for step in range(total_steps):
    # Base (hip) moves forward continuously, centered between feet horizontally
    base_trajectory[step, 0] = stride_length * step / total_steps + stride_length / 2
    base_trajectory[step, 1] = (foot1_y + foot2_y) / 2  # Centered between feet
    base_trajectory[step, 2] = hip_height
    
    com_trajectory[step, :] = base_trajectory[step, :]
    
    # Determine position in current cycle
    cycle_pos = (step % cycle_length) / cycle_length  # 0 to 1
    cycle_num = step // cycle_length
    
    # Foot 1: plants at beginning of each cycle, stays during stance, swings forward
    foot1_plant_x = stride_length * cycle_num + stride_length / 2
    
    if cycle_pos < 0.68:  # 68% stance
        # STANCE: Foot planted at fixed position for this cycle
        foot1_trajectory[step, 0] = foot1_plant_x
        foot1_trajectory[step, 1] = foot1_y
        foot1_trajectory[step, 2] = ground_z
    else:
        # SWING: Foot lifted and moves to next cycle's plant position
        swing_norm = (cycle_pos - 0.68) / 0.32
        next_plant_x = foot1_plant_x + stride_length
        
        # Interpolate smoothly from current plant to next plant
        foot1_trajectory[step, 0] = foot1_plant_x + (next_plant_x - foot1_plant_x) * swing_norm
        foot1_trajectory[step, 1] = foot1_y
        lift_height = (swing_z - ground_z) * np.sin(np.pi * swing_norm)
        foot1_trajectory[step, 2] = ground_z + lift_height
    
    # Foot 2: 180° out of phase with Foot 1
    foot2_phase = (cycle_pos + 0.5) % 1.0
    foot2_cycle = cycle_num if cycle_pos < 0.5 else cycle_num + 1
    foot2_plant_x = stride_length * (foot2_cycle) + stride_length / 2
    
    if foot2_phase < 0.68:  # 68% stance
        # STANCE: Foot planted at fixed position for this cycle
        foot2_trajectory[step, 0] = foot2_plant_x
        foot2_trajectory[step, 1] = foot2_y
        foot2_trajectory[step, 2] = ground_z
    else:
        # SWING: Foot lifted and moves to next cycle's plant position
        swing_norm = (foot2_phase - 0.68) / 0.32
        next_plant_x = foot2_plant_x + stride_length
        
        # Interpolate smoothly from current plant to next plant
        foot2_trajectory[step, 0] = foot2_plant_x + (next_plant_x - foot2_plant_x) * swing_norm
        foot2_trajectory[step, 1] = foot2_y
        lift_height = (swing_z - ground_z) * np.sin(np.pi * swing_norm)
        foot2_trajectory[step, 2] = ground_z + lift_height

# Validate
print("Trajectory statistics:")
print(f"Foot1 Z range: [{foot1_trajectory[:, 2].min():.6f}, {foot1_trajectory[:, 2].max():.6f}]")
print(f"Foot2 Z range: [{foot2_trajectory[:, 2].min():.6f}, {foot2_trajectory[:, 2].max():.6f}]")

foot1_contact = np.sum(np.abs(foot1_trajectory[:, 2] - ground_z) < 0.00001)
foot2_contact = np.sum(np.abs(foot2_trajectory[:, 2] - ground_z) < 0.00001)
print(f"Foot1 ground contact: {foot1_contact} steps ({100*foot1_contact/total_steps:.1f}%)")
print(f"Foot2 ground contact: {foot2_contact} steps ({100*foot2_contact/total_steps:.1f}%)")

# Save
np.save('base_trajectory.npy', base_trajectory)
np.save('foot1_trajectory.npy', foot1_trajectory)
np.save('foot2_trajectory.npy', foot2_trajectory)
np.save('com_trajectory.npy', com_trajectory)

print("Trajectories saved!")
