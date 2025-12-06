#!/usr/bin/env python3
"""
Generate Feasible Walking Trajectories

Creates trajectories within the robot's reachable workspace.
"""

import numpy as np
import mujoco

def generate_feasible_trajectories():
    """Generate trajectories within reachable workspace."""
    
    print("\n" + "="*80)
    print("GENERATING FEASIBLE WALKING TRAJECTORIES")
    print("="*80)
    
    # First, analyze workspace
    print("\nAnalyzing workspace...")
    model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
    data = mujoco.MjData(model)
    
    # Sample workspace with different joint angles
    hip_range = np.linspace(-np.pi/3, 0, 10)
    knee_range = np.linspace(-np.pi/2, 0, 10)
    ankle_range = np.linspace(0, np.pi/2, 10)
    
    positions = []
    for hip in hip_range:
        for knee in knee_range:
            for ankle in ankle_range:
                data.qpos[3:6] = [hip, knee, ankle]
                mujoco.mj_forward(model, data)
                positions.append(data.site_xpos[0].copy())
    
    positions = np.array(positions)
    
    # Determine safe workspace bounds
    x_min, x_max = positions[:, 0].min() * 1.1, positions[:, 0].max() * 0.9
    y_min, y_max = -0.015, -0.015  # Keep feet at lateral offset
    z_min, z_max = positions[:, 2].min() * 0.9, positions[:, 2].max() * 0.95
    
    print(f"\nSafe workspace bounds:")
    print(f"  X: {x_min*1000:.2f} to {x_max*1000:.2f} mm")
    print(f"  Y: {y_min*1000:.2f} to {y_max*1000:.2f} mm")
    print(f"  Z: {z_min*1000:.2f} to {z_max*1000:.2f} mm")
    
    # Generate trajectories within workspace
    NUM_STEPS = 400
    CYCLE_STEPS = 100
    STANCE_FRACTION = 0.60
    STANCE_STEPS = int(CYCLE_STEPS * STANCE_FRACTION)
    SWING_STEPS = CYCLE_STEPS - STANCE_STEPS
    
    # Reduced motion parameters
    STRIDE_LENGTH = 0.008  # Wider stride for more natural walking
    Z_MEAN = (z_min + z_max) / 2
    Z_SWING = (z_max - z_min) / 6  # Reduced swing amplitude for lower foot lift
    FOOT_SPACING = 0.020
    
    base_trajectory = np.zeros((NUM_STEPS, 3))
    foot1_trajectory = np.zeros((NUM_STEPS, 3))
    foot2_trajectory = np.zeros((NUM_STEPS, 3))
    
    print(f"\nGenerating feasible trajectories...")
    print(f"  Stride: {STRIDE_LENGTH*1000:.2f} mm")
    print(f"  Hip Z center: {Z_MEAN*1000:.2f} mm")
    
    for step in range(NUM_STEPS):
        # Base trajectory - hip positioned above feet (ahead of stance foot)
        cycle_pos = step % CYCLE_STEPS
        contact_cycle = step // CYCLE_STEPS
        
        # Hip is always ahead of stance foot by about half a stride
        hip_x_offset = STRIDE_LENGTH * 0.5
        base_trajectory[step, 0] = STRIDE_LENGTH * contact_cycle + hip_x_offset
        base_trajectory[step, 1] = 0.0
        base_trajectory[step, 2] = Z_MEAN  # Keep hip at center height
        
        # Left foot
        cycle_pos = step % CYCLE_STEPS
        
        if cycle_pos < STANCE_STEPS:
            # Stance: foot on ground at back position
            contact_cycle = step // CYCLE_STEPS
            foot1_trajectory[step, 0] = STRIDE_LENGTH * contact_cycle
            foot1_trajectory[step, 1] = -FOOT_SPACING
            foot1_trajectory[step, 2] = z_min
        else:
            # Swing: foot moves forward and up
            swing_progress = (cycle_pos - STANCE_STEPS) / SWING_STEPS
            contact_cycle = step // CYCLE_STEPS
            current_x = STRIDE_LENGTH * contact_cycle
            next_x = STRIDE_LENGTH * (contact_cycle + 1)
            
            foot1_trajectory[step, 0] = current_x + (next_x - current_x) * swing_progress
            foot1_trajectory[step, 1] = -FOOT_SPACING
            
            # Arc motion in Z
            lift = Z_SWING * np.sin(np.pi * swing_progress)
            foot1_trajectory[step, 2] = z_min + lift
        
        # Right foot (180° out of phase)
        cycle_pos2 = (step + CYCLE_STEPS // 2) % CYCLE_STEPS
        
        if cycle_pos2 < STANCE_STEPS:
            contact_cycle2 = (step + CYCLE_STEPS // 2) // CYCLE_STEPS
            foot2_trajectory[step, 0] = STRIDE_LENGTH * contact_cycle2
            foot2_trajectory[step, 1] = FOOT_SPACING
            foot2_trajectory[step, 2] = z_min
        else:
            swing_progress2 = (cycle_pos2 - STANCE_STEPS) / SWING_STEPS
            contact_cycle2 = (step + CYCLE_STEPS // 2) // CYCLE_STEPS
            current_x2 = STRIDE_LENGTH * contact_cycle2
            next_x2 = STRIDE_LENGTH * (contact_cycle2 + 1)
            
            foot2_trajectory[step, 0] = current_x2 + (next_x2 - current_x2) * swing_progress2
            foot2_trajectory[step, 1] = FOOT_SPACING
            
            lift2 = Z_SWING * np.sin(np.pi * swing_progress2)
            foot2_trajectory[step, 2] = z_min + lift2
    
    # Validation
    print(f"\n" + "-"*80)
    print("VALIDATION")
    print("-"*80)
    
    # Check if within workspace
    foot1_in_workspace = (
        (foot1_trajectory[:, 0] >= x_min) & (foot1_trajectory[:, 0] <= x_max) &
        (foot1_trajectory[:, 1] >= y_min) & (foot1_trajectory[:, 1] <= y_max) &
        (foot1_trajectory[:, 2] >= z_min) & (foot1_trajectory[:, 2] <= z_max)
    )
    
    foot2_in_workspace = (
        (foot2_trajectory[:, 0] >= x_min) & (foot2_trajectory[:, 0] <= x_max) &
        (foot2_trajectory[:, 1] >= y_min) & (foot2_trajectory[:, 1] <= y_max) &
        (foot2_trajectory[:, 2] >= z_min) & (foot2_trajectory[:, 2] <= z_max)
    )
    
    print(f"\nLeft foot in workspace: {np.sum(foot1_in_workspace)}/{len(foot1_in_workspace)}")
    print(f"Right foot in workspace: {np.sum(foot2_in_workspace)}/{len(foot2_in_workspace)}")
    
    # Save
    print(f"\n" + "-"*80)
    print("SAVING")
    print("-"*80)
    
    np.save("base_trajectory_feasible.npy", base_trajectory)
    np.save("foot1_trajectory_feasible.npy", foot1_trajectory)
    np.save("foot2_trajectory_feasible.npy", foot2_trajectory)
    
    print(f"✓ base_trajectory_feasible.npy")
    print(f"✓ foot1_trajectory_feasible.npy")
    print(f"✓ foot2_trajectory_feasible.npy")
    
    print(f"\nTrajectory Statistics:")
    print(f"  Base X: {base_trajectory[:, 0].min()*1000:.2f} to {base_trajectory[:, 0].max()*1000:.2f} mm")
    print(f"  Foot1 X: {foot1_trajectory[:, 0].min()*1000:.2f} to {foot1_trajectory[:, 0].max()*1000:.2f} mm")
    print(f"  Foot1 Z: {foot1_trajectory[:, 2].min()*1000:.2f} to {foot1_trajectory[:, 2].max()*1000:.2f} mm")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    generate_feasible_trajectories()
