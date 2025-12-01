#!/usr/bin/env python3
"""
Plot the generated humanoid walking trajectories
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load trajectories
print("Loading trajectories...")
base_traj = np.load("base_trajectory.npy")
com_traj = np.load("com_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

num_steps = len(com_traj)
time_steps = np.arange(num_steps)

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# ============ 3D PLOT: Full Trajectories ============
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot(base_traj[:, 0], base_traj[:, 1], base_traj[:, 2], 'k-', linewidth=2, label='Base')
ax1.plot(com_traj[:, 0], com_traj[:, 1], com_traj[:, 2], 'r-', linewidth=2, label='COM')
ax1.plot(foot1_traj[:, 0], foot1_traj[:, 1], foot1_traj[:, 2], 'b-', linewidth=1.5, label='Foot 1 (Left)')
ax1.plot(foot2_traj[:, 0], foot2_traj[:, 1], foot2_traj[:, 2], 'g-', linewidth=1.5, label='Foot 2 (Right)')
ax1.set_xlabel('X (forward) [m]')
ax1.set_ylabel('Y (lateral) [m]')
ax1.set_zlabel('Z (height) [m]')
ax1.set_title('3D Trajectory View')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ============ X POSITION: Forward Progression ============
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(time_steps, base_traj[:, 0], 'k-', linewidth=2, label='Base X')
ax2.plot(time_steps, com_traj[:, 0], 'r-', linewidth=2, label='COM X')
ax2.plot(time_steps, foot1_traj[:, 0], 'b--', linewidth=1.5, label='Foot1 X')
ax2.plot(time_steps, foot2_traj[:, 0], 'g--', linewidth=1.5, label='Foot2 X')
ax2.set_xlabel('Step')
ax2.set_ylabel('X Position [m]')
ax2.set_title('Forward Progression (X)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ============ Y POSITION: Lateral Motion ============
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(time_steps, base_traj[:, 1], 'k-', linewidth=2, label='Base Y')
ax3.plot(time_steps, com_traj[:, 1], 'r-', linewidth=2, label='COM Y')
ax3.plot(time_steps, foot1_traj[:, 1], 'b--', linewidth=1.5, label='Foot1 Y')
ax3.plot(time_steps, foot2_traj[:, 1], 'g--', linewidth=1.5, label='Foot2 Y')
ax3.set_xlabel('Step')
ax3.set_ylabel('Y Position [m]')
ax3.set_title('Lateral Motion (Y)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ============ Z POSITION: Height Profile ============
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(time_steps, base_traj[:, 2], 'k-', linewidth=2, label='Base Z')
ax4.plot(time_steps, com_traj[:, 2], 'r-', linewidth=2, label='COM Z')
ax4.plot(time_steps, foot1_traj[:, 2], 'b-', linewidth=1.5, label='Foot1 Z (Left)')
ax4.plot(time_steps, foot2_traj[:, 2], 'g-', linewidth=1.5, label='Foot2 Z (Right)')
ax4.axhline(y=0.212548, color='k', linestyle=':', alpha=0.5, label='Ground level')
ax4.set_xlabel('Step')
ax4.set_ylabel('Z Position [m]')
ax4.set_title('Height Profile (Z)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# ============ TOP VIEW (XY PLANE) ============
ax5 = fig.add_subplot(2, 3, 5)
# Draw walking path
ax5.plot(base_traj[:, 0], base_traj[:, 1], 'k-', linewidth=2, label='Base path')
ax5.plot(com_traj[:, 0], com_traj[:, 1], 'r-', linewidth=2, label='COM path')
ax5.plot(foot1_traj[:, 0], foot1_traj[:, 1], 'b-', linewidth=1, alpha=0.7, label='Foot1 path')
ax5.plot(foot2_traj[:, 0], foot2_traj[:, 1], 'g-', linewidth=1, alpha=0.7, label='Foot2 path')

# Mark start and end
ax5.plot(base_traj[0, 0], base_traj[0, 1], 'ko', markersize=10, label='Start')
ax5.plot(base_traj[-1, 0], base_traj[-1, 1], 'kx', markersize=12, markeredgewidth=2, label='End')

ax5.set_xlabel('X (forward) [m]')
ax5.set_ylabel('Y (lateral) [m]')
ax5.set_title('Top View (XY Plane)')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.axis('equal')

# ============ SIDE VIEW (XZ PLANE - Sagittal) ============
ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(base_traj[:, 0], base_traj[:, 2], 'k-', linewidth=2, label='Base')
ax6.plot(com_traj[:, 0], com_traj[:, 2], 'r-', linewidth=2, label='COM')
ax6.plot(foot1_traj[:, 0], foot1_traj[:, 2], 'b-', linewidth=1, alpha=0.7, label='Foot1')
ax6.plot(foot2_traj[:, 0], foot2_traj[:, 2], 'g-', linewidth=1, alpha=0.7, label='Foot2')
ax6.axhline(y=0.212548, color='k', linestyle=':', alpha=0.5, linewidth=2, label='Ground')
ax6.set_xlabel('X (forward) [m]')
ax6.set_ylabel('Z (height) [m]')
ax6.set_title('Side View (XZ Sagittal Plane)')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('trajectories_plot.png', dpi=150, bbox_inches='tight')
print("Saved: trajectories_plot.png")

# ============ DETAILED FOOT CLEARANCE ANALYSIS ============
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Foot 1 clearance over steps
foot1_clearance = foot1_traj[:, 2] - 0.212548
axes[0, 0].fill_between(time_steps, 0, foot1_clearance, alpha=0.3, color='blue', label='Foot 1 clearance')
axes[0, 0].plot(time_steps, foot1_clearance, 'b-', linewidth=2)
axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=1)
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('Clearance Above Ground [m]')
axes[0, 0].set_title('Foot 1 (Left) Ground Clearance')
axes[0, 0].grid(True, alpha=0.3)

# Foot 2 clearance over steps
foot2_clearance = foot2_traj[:, 2] - 0.212548
axes[0, 1].fill_between(time_steps, 0, foot2_clearance, alpha=0.3, color='green', label='Foot 2 clearance')
axes[0, 1].plot(time_steps, foot2_clearance, 'g-', linewidth=2)
axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=1)
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('Clearance Above Ground [m]')
axes[0, 1].set_title('Foot 2 (Right) Ground Clearance')
axes[0, 1].grid(True, alpha=0.3)

# Foot positions XZ - Left foot swing
axes[1, 0].plot(foot1_traj[:, 0], foot1_traj[:, 2], 'b-', linewidth=2)
axes[1, 0].plot(foot1_traj[0, 0], foot1_traj[0, 2], 'go', markersize=10, label='Start')
axes[1, 0].plot(foot1_traj[-1, 0], foot1_traj[-1, 2], 'ro', markersize=10, label='End')
# Color by phase
for i in range(0, num_steps-1, 10):
    axes[1, 0].plot(foot1_traj[i:i+2, 0], foot1_traj[i:i+2, 2], 'b-', alpha=0.3)
axes[1, 0].axhline(y=0.212548, color='k', linestyle=':', linewidth=2, label='Ground')
axes[1, 0].set_xlabel('X Position [m]')
axes[1, 0].set_ylabel('Z Height [m]')
axes[1, 0].set_title('Foot 1 (Left) Swing Trajectory (XZ)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Foot positions XZ - Right foot swing
axes[1, 1].plot(foot2_traj[:, 0], foot2_traj[:, 2], 'g-', linewidth=2)
axes[1, 1].plot(foot2_traj[0, 0], foot2_traj[0, 2], 'bo', markersize=10, label='Start')
axes[1, 1].plot(foot2_traj[-1, 0], foot2_traj[-1, 2], 'ro', markersize=10, label='End')
for i in range(0, num_steps-1, 10):
    axes[1, 1].plot(foot2_traj[i:i+2, 0], foot2_traj[i:i+2, 2], 'g-', alpha=0.3)
axes[1, 1].axhline(y=0.212548, color='k', linestyle=':', linewidth=2, label='Ground')
axes[1, 1].set_xlabel('X Position [m]')
axes[1, 1].set_ylabel('Z Height [m]')
axes[1, 1].set_title('Foot 2 (Right) Swing Trajectory (XZ)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('foot_clearance_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: foot_clearance_analysis.png")

# ============ PRINT TRAJECTORY STATISTICS ============
print("\n" + "="*80)
print("TRAJECTORY STATISTICS")
print("="*80)

print("\nBase Trajectory:")
print(f"  X range: {base_traj[:, 0].min():.4f} to {base_traj[:, 0].max():.4f} m (stride: {base_traj[:, 0].max() - base_traj[:, 0].min():.4f} m)")
print(f"  Y range: {base_traj[:, 1].min():.4f} to {base_traj[:, 1].max():.4f} m")
print(f"  Z range: {base_traj[:, 2].min():.6f} to {base_traj[:, 2].max():.6f} m (fixed)")

print("\nCOM Trajectory:")
print(f"  X range: {com_traj[:, 0].min():.4f} to {com_traj[:, 0].max():.4f} m")
print(f"  Y range: {com_traj[:, 1].min():.6f} to {com_traj[:, 1].max():.6f} m (centered)")
print(f"  Z range: {com_traj[:, 2].min():.6f} to {com_traj[:, 2].max():.6f} m (bobbing: {com_traj[:, 2].max() - com_traj[:, 2].min():.6f} m)")

print("\nFoot 1 (Left) Trajectory:")
print(f"  X range: {foot1_traj[:, 0].min():.4f} to {foot1_traj[:, 0].max():.4f} m")
print(f"  Y: {foot1_traj[:, 1].min():.4f} m (constant lateral)")
print(f"  Z range: {foot1_traj[:, 2].min():.6f} to {foot1_traj[:, 2].max():.6f} m")
print(f"  Max clearance: {(foot1_traj[:, 2].max() - 0.212548)*1000:.2f} mm")

print("\nFoot 2 (Right) Trajectory:")
print(f"  X range: {foot2_traj[:, 0].min():.4f} to {foot2_traj[:, 0].max():.4f} m")
print(f"  Y: {foot2_traj[:, 1].max():.4f} m (constant lateral)")
print(f"  Z range: {foot2_traj[:, 2].min():.6f} to {foot2_traj[:, 2].max():.6f} m")
print(f"  Max clearance: {(foot2_traj[:, 2].max() - 0.212548)*1000:.2f} mm")

print("\nGait Characteristics:")
print(f"  Total steps: {num_steps}")
print(f"  Forward stride: {base_traj[-1, 0] - base_traj[0, 0]:.4f} m")
print(f"  Lateral step width (L-R): {abs(foot1_traj[0, 1] - foot2_traj[0, 1]):.4f} m")
print(f"  Max foot swing height: {(foot1_traj[:, 2].max() - foot1_traj[:, 2].min())*1000:.2f} mm")

print("\n" + "="*80)

plt.show()
