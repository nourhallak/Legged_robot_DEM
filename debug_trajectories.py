"""Debug script to inspect trajectory files"""
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load and inspect trajectory files
com_traj = np.load(os.path.join(script_dir, "com_trajectory.npy"))
foot1_traj = np.load(os.path.join(script_dir, "foot1_trajectory.npy"))
foot2_traj = np.load(os.path.join(script_dir, "foot2_trajectory.npy"))

print("=== TRAJECTORY INSPECTION ===\n")

print("COM Trajectory:")
print(f"  Shape: {com_traj.shape}")
print(f"  First 3 frames:\n{com_traj[:3]}")
print(f"  Last 3 frames:\n{com_traj[-3:]}")
print(f"  Min/Max: {com_traj.min():.6f} / {com_traj.max():.6f}")
print(f"  Range: {(com_traj.max() - com_traj.min()):.6f}")
print(f"  Unique frames: {len(np.unique(com_traj, axis=0))}")

print("\nFoot 1 Trajectory:")
print(f"  Shape: {foot1_traj.shape}")
print(f"  First 3 frames:\n{foot1_traj[:3]}")
print(f"  Last 3 frames:\n{foot1_traj[-3:]}")
print(f"  Min/Max: {foot1_traj.min():.6f} / {foot1_traj.max():.6f}")
print(f"  Range: {(foot1_traj.max() - foot1_traj.min()):.6f}")
print(f"  Unique frames: {len(np.unique(foot1_traj, axis=0))}")

print("\nFoot 2 Trajectory:")
print(f"  Shape: {foot2_traj.shape}")
print(f"  First 3 frames:\n{foot2_traj[:3]}")
print(f"  Last 3 frames:\n{foot2_traj[-3:]}")
print(f"  Min/Max: {foot2_traj.min():.6f} / {foot2_traj.max():.6f}")
print(f"  Range: {(foot2_traj.max() - foot2_traj.min()):.6f}")
print(f"  Unique frames: {len(np.unique(foot2_traj, axis=0))}")

# Check if trajectories change at all
if len(np.unique(com_traj, axis=0)) == 1:
    print("\n⚠️  WARNING: COM trajectory has only 1 unique frame (static!)")
if len(np.unique(foot1_traj, axis=0)) == 1:
    print("⚠️  WARNING: Foot 1 trajectory has only 1 unique frame (static!)")
if len(np.unique(foot2_traj, axis=0)) == 1:
    print("⚠️  WARNING: Foot 2 trajectory has only 1 unique frame (static!)")
