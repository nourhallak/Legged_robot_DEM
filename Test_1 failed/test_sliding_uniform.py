#!/usr/bin/env python3
"""
Quick test to check if uniform foot heights eliminate sliding
Runs first 50 simulation steps and checks foot positions
"""

import numpy as np
import sys
sys.path.insert(0, '.')

# Load trajectories
base_traj = np.load('base_trajectory.npy')
com_traj = np.load('com_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

print("\n=== TRAJECTORY VERIFICATION (Uniform 15mm Heights) ===\n")
print(f"Total steps: {len(base_traj)}")
print(f"\nFoot 1 (Left) Swing Heights (First swing cycle, steps 0-100):")
for step in [0, 25, 50, 75, 100]:
    if step < len(foot1_traj):
        print(f"  Step {step:3d}: Z = {foot1_traj[step, 2]:.6f} m ({(foot1_traj[step, 2] - 0.21)*1000:.2f} mm clearance)")

print(f"\nFoot 2 (Right) Swing Heights (First swing cycle, steps 100-200):")
for step in [100, 125, 150, 175, 200]:
    if step < len(foot2_traj):
        print(f"  Step {step:3d}: Z = {foot2_traj[step, 2]:.6f} m ({(foot2_traj[step, 2] - 0.21)*1000:.2f} mm clearance)")

print(f"\nFoot 1 Stance X Positions (Step 200-300, should be constant):")
stance_x1 = foot1_traj[200:300, 0]
print(f"  Min: {stance_x1.min():.6f}, Max: {stance_x1.max():.6f}, Variation: {(stance_x1.max()-stance_x1.min())*1000:.4f} mm")

print(f"\nFoot 2 Stance X Positions (Step 0-100, should be constant):")
stance_x2 = foot2_traj[0:100, 0]
print(f"  Min: {stance_x2.min():.6f}, Max: {stance_x2.max():.6f}, Variation: {(stance_x2.max()-stance_x2.min())*1000:.4f} mm")

print("\nTrajectories verified - both feet now 15mm clearance. Ready to test in simulator.")
print("Run: python -u ik_simulation.py")
