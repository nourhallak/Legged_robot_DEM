"""Adjust trajectory to match robot geometry"""
import numpy as np

# Load original trajectories
com_traj = np.load("com_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

print("Original trajectories:")
print(f"  COM Z: {com_traj[:, 2].min():.4f} to {com_traj[:, 2].max():.4f}")
print(f"  Foot1 Z: {foot1_traj[:, 2].min():.4f} to {foot1_traj[:, 2].max():.4f}")
print(f"  Foot2 Z: {foot2_traj[:, 2].min():.4f} to {foot2_traj[:, 2].max():.4f}")

# Offset to make feet reach ground
# Robot feet are ~0.012m above z=0 when hip is at z=0
# So we need to shift everything DOWN by 0.01m
z_offset = -0.012

com_traj_adjusted = com_traj.copy()
foot1_traj_adjusted = foot1_traj.copy()
foot2_traj_adjusted = foot2_traj.copy()

com_traj_adjusted[:, 2] += z_offset
foot1_traj_adjusted[:, 2] += z_offset
foot2_traj_adjusted[:, 2] += z_offset

print("\nAdjusted trajectories (z-offset = -0.012):")
print(f"  COM Z: {com_traj_adjusted[:, 2].min():.4f} to {com_traj_adjusted[:, 2].max():.4f}")
print(f"  Foot1 Z: {foot1_traj_adjusted[:, 2].min():.4f} to {foot1_traj_adjusted[:, 2].max():.4f}")
print(f"  Foot2 Z: {foot2_traj_adjusted[:, 2].min():.4f} to {foot2_traj_adjusted[:, 2].max():.4f}")

# Save adjusted trajectories
np.save("com_trajectory.npy", com_traj_adjusted)
np.save("foot1_trajectory.npy", foot1_traj_adjusted)
np.save("foot2_trajectory.npy", foot2_traj_adjusted)

print("\n✓ Trajectories saved (adjusted)")
