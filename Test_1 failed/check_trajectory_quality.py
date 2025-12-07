#!/usr/bin/env python3
"""
Check trajectory quality for vibration/noise
"""
import numpy as np
import matplotlib.pyplot as plt

# Load trajectories
hip_traj = np.load('hip_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

print("\n" + "="*80)
print("TRAJECTORY QUALITY ANALYSIS")
print("="*80)

print(f"\nTrajectory shapes:")
print(f"  Hip: {hip_traj.shape}")
print(f"  Foot1: {foot1_traj.shape}")
print(f"  Foot2: {foot2_traj.shape}")

# Check for NaN or Inf
print(f"\nNaN/Inf check:")
print(f"  Hip NaN: {np.isnan(hip_traj).sum()}, Inf: {np.isinf(hip_traj).sum()}")
print(f"  Foot1 NaN: {np.isnan(foot1_traj).sum()}, Inf: {np.isinf(foot1_traj).sum()}")
print(f"  Foot2 NaN: {np.isnan(foot2_traj).sum()}, Inf: {np.isinf(foot2_traj).sum()}")

# Compute velocity (first derivative)
hip_vel = np.diff(hip_traj, axis=0)
foot1_vel = np.diff(foot1_traj, axis=0)
foot2_vel = np.diff(foot2_traj, axis=0)

print(f"\nVelocity statistics (mm/step):")
print(f"  Hip X: mean={np.mean(np.abs(hip_vel[:, 0])):.3f}, max={np.max(np.abs(hip_vel[:, 0])):.3f}")
print(f"  Hip Y: mean={np.mean(np.abs(hip_vel[:, 1])):.3f}, max={np.max(np.abs(hip_vel[:, 1])):.3f}")
print(f"  Hip Z: mean={np.mean(np.abs(hip_vel[:, 2])):.3f}, max={np.max(np.abs(hip_vel[:, 2])):.3f}")

# Compute acceleration (second derivative)
hip_acc = np.diff(hip_vel, axis=0)
foot1_acc = np.diff(foot1_vel, axis=0)
foot2_acc = np.diff(foot2_vel, axis=0)

print(f"\nAcceleration statistics (mm/step²):")
print(f"  Hip X: mean={np.mean(np.abs(hip_acc[:, 0])):.3f}, max={np.max(np.abs(hip_acc[:, 0])):.3f}")
print(f"  Hip Y: mean={np.mean(np.abs(hip_acc[:, 1])):.3f}, max={np.max(np.abs(hip_acc[:, 1])):.3f}")
print(f"  Hip Z: mean={np.mean(np.abs(hip_acc[:, 2])):.3f}, max={np.max(np.abs(hip_acc[:, 2])):.3f}")

print(f"\n  Foot1 X: mean={np.mean(np.abs(foot1_acc[:, 0])):.3f}, max={np.max(np.abs(foot1_acc[:, 0])):.3f}")
print(f"  Foot1 Y: mean={np.mean(np.abs(foot1_acc[:, 1])):.3f}, max={np.max(np.abs(foot1_acc[:, 1])):.3f}")
print(f"  Foot1 Z: mean={np.mean(np.abs(foot1_acc[:, 2])):.3f}, max={np.max(np.abs(foot1_acc[:, 2])):.3f}")

print(f"\n  Foot2 X: mean={np.mean(np.abs(foot2_acc[:, 0])):.3f}, max={np.max(np.abs(foot2_acc[:, 0])):.3f}")
print(f"  Foot2 Y: mean={np.mean(np.abs(foot2_acc[:, 1])):.3f}, max={np.max(np.abs(foot2_acc[:, 1])):.3f}")
print(f"  Foot2 Z: mean={np.mean(np.abs(foot2_acc[:, 2])):.3f}, max={np.max(np.abs(foot2_acc[:, 2])):.3f}")

# Check for discontinuities (large jumps)
print(f"\nDiscontinuity check (step-to-step jumps > 5mm):")
hip_jumps = np.linalg.norm(hip_vel, axis=1)
foot1_jumps = np.linalg.norm(foot1_vel, axis=1)
foot2_jumps = np.linalg.norm(foot2_vel, axis=1)

hip_disc = np.sum(hip_jumps > 5)
foot1_disc = np.sum(foot1_jumps > 5)
foot2_disc = np.sum(foot2_jumps > 5)

print(f"  Hip: {hip_disc} jumps > 5mm (max: {np.max(hip_jumps):.2f}mm)")
print(f"  Foot1: {foot1_disc} jumps > 5mm (max: {np.max(foot1_jumps):.2f}mm)")
print(f"  Foot2: {foot2_disc} jumps > 5mm (max: {np.max(foot2_jumps):.2f}mm)")

# Plot trajectories
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle('Trajectory Analysis', fontsize=14)

# Hip trajectory
axes[0, 0].plot(hip_traj[:, 0])
axes[0, 0].set_title('Hip X')
axes[0, 0].set_ylabel('Position (mm)')
axes[0, 1].plot(hip_traj[:, 1])
axes[0, 1].set_title('Hip Y')
axes[0, 2].plot(hip_traj[:, 2])
axes[0, 2].set_title('Hip Z')

# Foot1 trajectory
axes[1, 0].plot(foot1_traj[:, 0])
axes[1, 0].set_title('Foot1 X')
axes[1, 0].set_ylabel('Position (mm)')
axes[1, 1].plot(foot1_traj[:, 1])
axes[1, 1].set_title('Foot1 Y')
axes[1, 2].plot(foot1_traj[:, 2])
axes[1, 2].set_title('Foot1 Z')

# Foot2 trajectory
axes[2, 0].plot(foot2_traj[:, 0])
axes[2, 0].set_title('Foot2 X')
axes[2, 0].set_xlabel('Step')
axes[2, 0].set_ylabel('Position (mm)')
axes[2, 1].plot(foot2_traj[:, 1])
axes[2, 1].set_title('Foot2 Y')
axes[2, 1].set_xlabel('Step')
axes[2, 2].plot(foot2_traj[:, 2])
axes[2, 2].set_title('Foot2 Z')
axes[2, 2].set_xlabel('Step')

plt.tight_layout()
plt.savefig('trajectory_quality.png', dpi=100)
print(f"\nPlot saved to trajectory_quality.png")

# Check if trajectories are repeating/looping correctly
print(f"\nLoop continuity check:")
print(f"  Hip start: {hip_traj[0]}")
print(f"  Hip end:   {hip_traj[-1]}")
print(f"  Foot1 start: {foot1_traj[0]}")
print(f"  Foot1 end:   {foot1_traj[-1]}")
print(f"  Foot2 start: {foot2_traj[0]}")
print(f"  Foot2 end:   {foot2_traj[-1]}")

print("\n" + "="*80)
