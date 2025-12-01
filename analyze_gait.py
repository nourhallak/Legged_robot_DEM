"""Analyze if trajectories represent walking motion"""
import numpy as np

com_traj = np.load("com_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

print("=== WALKING MOTION ANALYSIS ===\n")

# For walking, we expect feet to alternate between:
# - Stance phase: y = constant (touching ground)
# - Swing phase: y varies + z rises above ground

print("Foot 1 trajectory analysis:")
print(f"  X range: {foot1_traj[:, 0].min():.4f} to {foot1_traj[:, 0].max():.4f} (forward motion: {foot1_traj[:, 0].max() - foot1_traj[:, 0].min():.4f})")
print(f"  Y range: {foot1_traj[:, 1].min():.4f} to {foot1_traj[:, 1].max():.4f} (lateral)")
print(f"  Z range: {foot1_traj[:, 2].min():.4f} to {foot1_traj[:, 2].max():.4f} (height)")

# Count unique z values (if only 0, it's on ground entire time)
unique_z1 = len(np.unique(np.round(foot1_traj[:, 2], 4)))
print(f"  Unique Z heights: {unique_z1}")

print("\nFoot 2 trajectory analysis:")
print(f"  X range: {foot2_traj[:, 0].min():.4f} to {foot2_traj[:, 0].max():.4f} (forward motion: {foot2_traj[:, 0].max() - foot2_traj[:, 0].min():.4f})")
print(f"  Y range: {foot2_traj[:, 1].min():.4f} to {foot2_traj[:, 1].max():.4f} (lateral)")
print(f"  Z range: {foot2_traj[:, 2].min():.4f} to {foot2_traj[:, 2].max():.4f} (height)")

unique_z2 = len(np.unique(np.round(foot2_traj[:, 2], 4)))
print(f"  Unique Z heights: {unique_z2}")

print("\nCOM trajectory analysis:")
print(f"  X range: {com_traj[:, 0].min():.4f} to {com_traj[:, 0].max():.4f}")
print(f"  Y range: {com_traj[:, 1].min():.4f} to {com_traj[:, 1].max():.4f}")
print(f"  Z range: {com_traj[:, 0].min():.4f} to {com_traj[:, 0].max():.4f}")
print(f"  COM path length: {np.linalg.norm(np.diff(com_traj, axis=0)).sum():.4f}")

print("\n=== MOTION TYPE ASSESSMENT ===")
if unique_z1 < 5 and unique_z2 < 5:
    print("⚠️  Both feet stay at constant height - NOT a walking motion!")
    print("   This looks like FORWARD SLIDING, not leg motion.")
elif foot1_traj[:, 2].max() - foot1_traj[:, 2].min() > 0.05 or foot2_traj[:, 2].max() - foot2_traj[:, 2].min() > 0.05:
    print("✓ Feet lift off ground - This IS a walking motion")
    print("  Feet swing up and down as expected in gait")
else:
    print("? Feet have minimal height variation")

# Check alternation pattern
print("\n=== GAIT PHASE ANALYSIS ===")
foot1_z_diffs = np.diff(foot1_traj[:, 2])
foot2_z_diffs = np.diff(foot2_traj[:, 2])

# Find peaks in z position (swing phase)
swing1_count = np.sum(foot1_z_diffs > 0.001)
swing2_count = np.sum(foot2_z_diffs > 0.001)

print(f"Foot 1 rising steps: {swing1_count}/{len(foot1_z_diffs)}")
print(f"Foot 2 rising steps: {swing2_count}/{len(foot2_z_diffs)}")

# Show some sample frames
print("\n=== SAMPLE TRAJECTORY FRAMES ===")
print("Frame  | Foot1 Z | Foot2 Z | COM X")
for i in [0, 50, 100, 150, 200, 250, 300, 350, 399]:
    print(f"{i:3d}    | {foot1_traj[i, 2]:.4f}  | {foot2_traj[i, 2]:.4f}  | {com_traj[i, 0]:.4f}")
