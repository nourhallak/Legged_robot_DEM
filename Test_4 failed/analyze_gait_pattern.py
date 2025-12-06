#!/usr/bin/env python3
"""
Analyze gait pattern to check if both legs are moving
"""

import numpy as np
import matplotlib.pyplot as plt

base_traj = np.load("base_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

print("="*80)
print("GAIT ANALYSIS - CHECK BOTH LEGS")
print("="*80)

# Check foot1 motion
foot1_x_range = np.max(foot1_traj[:, 0]) - np.min(foot1_traj[:, 0])
foot1_z_range = np.max(foot1_traj[:, 2]) - np.min(foot1_traj[:, 2])

print(f"\nFoot1 (Left):")
print(f"  X range: {foot1_x_range*1000:.2f}mm")
print(f"  Z range: {foot1_z_range*1000:.2f}mm (lift height)")
print(f"  Min X: {np.min(foot1_traj[:, 0]):.5f}m, Max X: {np.max(foot1_traj[:, 0]):.5f}m")
print(f"  Min Z: {np.min(foot1_traj[:, 2]):.5f}m, Max Z: {np.max(foot1_traj[:, 2]):.5f}m")

# Check foot2 motion
foot2_x_range = np.max(foot2_traj[:, 0]) - np.min(foot2_traj[:, 0])
foot2_z_range = np.max(foot2_traj[:, 2]) - np.min(foot2_traj[:, 2])

print(f"\nFoot2 (Right):")
print(f"  X range: {foot2_x_range*1000:.2f}mm")
print(f"  Z range: {foot2_z_range*1000:.2f}mm (lift height)")
print(f"  Min X: {np.min(foot2_traj[:, 0]):.5f}m, Max X: {np.max(foot2_traj[:, 0]):.5f}m")
print(f"  Min Z: {np.min(foot2_traj[:, 2]):.5f}m, Max Z: {np.max(foot2_traj[:, 2]):.5f}m")

# Sample a gait cycle
print(f"\n" + "="*80)
print("GAIT CYCLE SAMPLE (first 100 steps)")
print("="*80)

cycle_len = 100
for t in range(0, cycle_len, 10):
    foot1_z = foot1_traj[t, 2]
    foot2_z = foot2_traj[t, 2]
    foot1_x = foot1_traj[t, 0]
    foot2_x = foot2_traj[t, 0]
    
    foot1_phase = "SWING" if foot1_z > 0.431 else "STANCE"
    foot2_phase = "SWING" if foot2_z > 0.431 else "STANCE"
    
    print(f"Step {t:3d}: Foot1 {foot1_phase} (Z={foot1_z:.4f}m, X={foot1_x:.5f}m) | Foot2 {foot2_phase} (Z={foot2_z:.4f}m, X={foot2_x:.5f}m)")

# Plot trajectories
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# X trajectories
axes[0, 0].plot(foot1_traj[:, 0], label='Foot1 (Left)', linewidth=2)
axes[0, 0].plot(foot2_traj[:, 0], label='Foot2 (Right)', linewidth=2)
axes[0, 0].set_ylabel('X Position (m)')
axes[0, 0].set_title('Foot X Position Over Gait')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Z trajectories
axes[0, 1].plot(foot1_traj[:, 2], label='Foot1 (Left)', linewidth=2)
axes[0, 1].plot(foot2_traj[:, 2], label='Foot2 (Right)', linewidth=2)
axes[0, 1].set_ylabel('Z Position (m)')
axes[0, 1].set_title('Foot Z Position (Height) Over Gait')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Phase diagram
t_cycle = np.arange(100)
foot1_phase = np.where(foot1_traj[:100, 2] > 0.431, 1, 0)
foot2_phase = np.where(foot2_traj[:100, 2] > 0.431, 1, 0)

axes[1, 0].fill_between(t_cycle, 0, foot1_phase, alpha=0.5, label='Foot1 Swing')
axes[1, 0].fill_between(t_cycle, -1, -foot2_phase, alpha=0.5, label='Foot2 Swing')
axes[1, 0].set_ylabel('Phase')
axes[1, 0].set_xlabel('Step in Cycle')
axes[1, 0].set_title('Swing Phase (1=swing, 0=stance)')
axes[1, 0].set_ylim(-1.5, 1.5)
axes[1, 0].legend()
axes[1, 0].grid(True)

# Y offset check
axes[1, 1].scatter(foot1_traj[:, 1], foot1_traj[:, 0], alpha=0.5, s=10, label='Foot1 (Left)')
axes[1, 1].scatter(foot2_traj[:, 1], foot2_traj[:, 0], alpha=0.5, s=10, label='Foot2 (Right)')
axes[1, 1].set_xlabel('Y Position (m)')
axes[1, 1].set_ylabel('X Position (m)')
axes[1, 1].set_title('Foot Positions in X-Y Plane')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('gait_analysis.png', dpi=150)
print(f"\n[OK] Saved gait_analysis.png")
