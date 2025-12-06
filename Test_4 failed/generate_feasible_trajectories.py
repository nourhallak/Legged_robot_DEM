#!/usr/bin/env python3
"""
Generate walking trajectories - BASE + FEET in WORLD coordinates

Key: Base moves forward CONTINUOUSLY. Feet plant at discrete positions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print("BIPEDAL WALKING TRAJECTORY GENERATOR - WORLD COORDINATES")
print("="*80)

stride_length = 0.005   # 5mm forward per cycle
cycle_steps = 100       
stance_fraction = 0.70  # 70% stance, 30% swing (more stable walking)
swing_clearance = 0.004 # 4mm clearance (smaller, smoother)
hip_height = 0.43       # Higher hip to reach ground with feet

left_foot_offset_y = -0.004   
right_foot_offset_y = 0.002   
foot_ground_z = 0.431         # Slightly higher than actual ground
foot_swing_z = 0.431 + 0.008  # Height during swing (ground + clearance)

num_cycles = 4
num_steps = cycle_steps * num_cycles

print(f"\nTrajectory Parameters:")
print(f"  Stride length: {stride_length*1000:.1f} mm")
print(f"  Cycle steps: {cycle_steps}")
print(f"  Stance fraction: {stance_fraction:.2f}")
print(f"  Swing clearance: {swing_clearance*1000:.1f} mm")
print(f"  Hip height: {hip_height*1000:.1f} mm")
print(f"  Foot ground Z: {foot_ground_z*1000:.1f} mm")
print(f"  Foot swing Z: {foot_swing_z*1000:.1f} mm")
print(f"  Total steps: {num_steps}")

# ============================================================================
# BASE TRAJECTORY - moves forward continuously
# ============================================================================

print("\nGenerating base trajectory...")

base_trajectory = np.zeros((num_steps, 3))

for step in range(num_steps):
    # Base moves forward gradually
    base_trajectory[step, 0] = (step / num_steps) * stride_length * num_cycles
    base_trajectory[step, 1] = 0.0
    base_trajectory[step, 2] = hip_height

print(f"  Base X range: [{base_trajectory[:,0].min():.4f}, {base_trajectory[:,0].max():.4f}]")
print(f"  Base Y range: [{base_trajectory[:,1].min():.4f}, {base_trajectory[:,1].max():.4f}]")
print(f"  Base Z range: [{base_trajectory[:,2].min():.4f}, {base_trajectory[:,2].max():.4f}]")

# ============================================================================
# FOOT TRAJECTORIES - IN WORLD COORDINATES (feet plant ahead of base)
# ============================================================================

print("\nGenerating left foot trajectory (world coordinates)...")

foot1_trajectory = np.zeros((num_steps, 3))

for step in range(num_steps):
    # Base position at this step
    base_x = (step / num_steps) * stride_length * num_cycles
    
    t = (step % cycle_steps) / cycle_steps
    
    if t < stance_fraction:
        # STANCE: foot planted on ground
        # Keep foot at a fixed X position (trailing behind body)
        foot1_trajectory[step, 0] = base_x - stride_length * 0.3
        foot1_trajectory[step, 1] = left_foot_offset_y  # Fixed Y offset
        foot1_trajectory[step, 2] = foot_ground_z
    else:
        # SWING: lift foot and swing forward
        t_swing = (t - stance_fraction) / (1.0 - stance_fraction)
        
        # Simple forward swing: from current position to ahead of base
        swing_start_x = base_x - stride_length * 0.3
        swing_end_x = base_x + stride_length * 0.3
        
        swing_x = swing_start_x + (swing_end_x - swing_start_x) * t_swing
        
        # Parabolic lift during swing
        swing_height = foot_ground_z + swing_clearance * np.sin(np.pi * t_swing)
        
        foot1_trajectory[step, 0] = swing_x
        foot1_trajectory[step, 1] = left_foot_offset_y  # Keep Y fixed
        foot1_trajectory[step, 2] = swing_height

print(f"  Foot1 X range (world): [{foot1_trajectory[:,0].min():.4f}, {foot1_trajectory[:,0].max():.4f}]")
print(f"  Foot1 Y range (world): [{foot1_trajectory[:,1].min():.4f}, {foot1_trajectory[:,1].max():.4f}]")
print(f"  Foot1 Z range (world): [{foot1_trajectory[:,2].min():.4f}, {foot1_trajectory[:,2].max():.4f}]")

print("\nGenerating right foot trajectory (world coordinates)...")

foot2_trajectory = np.zeros((num_steps, 3))

for step in range(num_steps):
    # Base position at this step
    base_x = (step / num_steps) * stride_length * num_cycles
    
    t = (step % cycle_steps) / cycle_steps
    
    # 180-degree out of phase: foot2 swings when foot1 stands, stands when foot1 swings
    t_phase = (t + 0.5) % 1.0
    
    if t_phase < stance_fraction:
        # STANCE: foot planted on ground
        # Keep foot at a fixed X position (trailing behind body)
        foot2_trajectory[step, 0] = base_x - stride_length * 0.3
        foot2_trajectory[step, 1] = right_foot_offset_y  # Fixed Y offset
        foot2_trajectory[step, 2] = foot_ground_z
    else:
        # SWING: lift foot and swing forward
        t_swing = (t_phase - stance_fraction) / (1.0 - stance_fraction)
        
        # Simple forward swing: from current position to ahead of base
        swing_start_x = base_x - stride_length * 0.3
        swing_end_x = base_x + stride_length * 0.3
        
        swing_x = swing_start_x + (swing_end_x - swing_start_x) * t_swing
        
        # Parabolic lift during swing
        swing_height = foot_ground_z + swing_clearance * np.sin(np.pi * t_swing)
        
        foot2_trajectory[step, 0] = swing_x
        foot2_trajectory[step, 1] = right_foot_offset_y  # Keep Y fixed
        foot2_trajectory[step, 2] = swing_height

print(f"  Foot2 X range (world): [{foot2_trajectory[:,0].min():.4f}, {foot2_trajectory[:,0].max():.4f}]")
print(f"  Foot2 Y range (world): [{foot2_trajectory[:,1].min():.4f}, {foot2_trajectory[:,1].max():.4f}]")
print(f"  Foot2 Z range (world): [{foot2_trajectory[:,2].min():.4f}, {foot2_trajectory[:,2].max():.4f}]")

# ============================================================================
# VALIDATION
# ============================================================================

print("\nValidating trajectories...")

# Check smoothness (limited acceleration)
base_vel = np.linalg.norm(np.diff(base_trajectory, axis=0), axis=1)
foot1_vel = np.linalg.norm(np.diff(foot1_trajectory, axis=0), axis=1)
foot2_vel = np.linalg.norm(np.diff(foot2_trajectory, axis=0), axis=1)

print(f"  Base velocity (max): {base_vel.max():.6f} m/step")
print(f"  Foot1 velocity (max): {foot1_vel.max():.6f} m/step")
print(f"  Foot2 velocity (max): {foot2_vel.max():.6f} m/step")

# Check contact (feet should touch ground during stance)
ground_z = 0.41  # Approximate ground level
foot1_contact = foot1_trajectory[:, 2] <= ground_z + 0.001
foot2_contact = foot2_trajectory[:, 2] <= ground_z + 0.001

print(f"  Foot1 ground contact steps: {foot1_contact.sum()} / {len(foot1_contact)}")
print(f"  Foot2 ground contact steps: {foot2_contact.sum()} / {len(foot2_contact)}")

# ============================================================================
# SAVE TRAJECTORIES
# ============================================================================

print("\nSaving trajectories...")
np.save("base_trajectory.npy", base_trajectory)
np.save("foot1_trajectory.npy", foot1_trajectory)
np.save("foot2_trajectory.npy", foot2_trajectory)
print("  [OK] Saved base_trajectory.npy")
print("  [OK] Saved foot1_trajectory.npy")
print("  [OK] Saved foot2_trajectory.npy")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating plots...")

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle("Bipedal Walking Trajectories (Feasible Version)", fontsize=14)

# Base trajectory
axes[0, 0].plot(base_trajectory[:, 0], 'b-', linewidth=2)
axes[0, 0].set_ylabel("Base X (m)")
axes[0, 0].grid(True)

axes[0, 1].plot(base_trajectory[:, 1], 'b-', linewidth=2)
axes[0, 1].set_ylabel("Base Y (m)")
axes[0, 1].grid(True)

axes[0, 2].plot(base_trajectory[:, 2], 'b-', linewidth=2)
axes[0, 2].set_ylabel("Base Z (m)")
axes[0, 2].grid(True)
axes[0, 2].axhline(y=hip_height, color='b', linestyle='--', alpha=0.5, label='nominal')
axes[0, 2].legend()

# Foot 1 trajectory
axes[1, 0].plot(foot1_trajectory[:, 0], 'r-', linewidth=2)
axes[1, 0].set_ylabel("Foot1 X (m)")
axes[1, 0].grid(True)

axes[1, 1].plot(foot1_trajectory[:, 1], 'r-', linewidth=2)
axes[1, 1].set_ylabel("Foot1 Y (m)")
axes[1, 1].grid(True)

axes[1, 2].plot(foot1_trajectory[:, 2], 'r-', linewidth=2)
axes[1, 2].set_ylabel("Foot1 Z (m)")
axes[1, 2].grid(True)
axes[1, 2].axhline(y=foot_ground_z, color='r', linestyle='--', alpha=0.5, label='ground')
axes[1, 2].legend()

# Foot 2 trajectory
axes[2, 0].plot(foot2_trajectory[:, 0], 'g-', linewidth=2)
axes[2, 0].set_ylabel("Foot2 X (m)")
axes[2, 0].set_xlabel("Step #")
axes[2, 0].grid(True)

axes[2, 1].plot(foot2_trajectory[:, 1], 'g-', linewidth=2)
axes[2, 1].set_ylabel("Foot2 Y (m)")
axes[2, 1].set_xlabel("Step #")
axes[2, 1].grid(True)

axes[2, 2].plot(foot2_trajectory[:, 2], 'g-', linewidth=2)
axes[2, 2].set_ylabel("Foot2 Z (m)")
axes[2, 2].set_xlabel("Step #")
axes[2, 2].grid(True)
axes[2, 2].axhline(y=foot_ground_z, color='g', linestyle='--', alpha=0.5, label='ground')
axes[2, 2].legend()

plt.tight_layout()
plt.savefig("walking_trajectories_feasible.png", dpi=150, bbox_inches='tight')
print("  [OK] Saved walking_trajectories_feasible.png")
plt.close()

print("\n" + "="*80)
print("Trajectory generation complete!")
print("="*80)
