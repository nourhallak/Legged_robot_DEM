#!/usr/bin/env python3
"""Visualize the realistic walking trajectories for the robot."""

import numpy as np
import matplotlib.pyplot as plt

# Load trajectories
times = np.load('traj_times.npy')
left_foot = np.load('traj_left_foot.npy')
right_foot = np.load('traj_right_foot.npy')
base_pos = np.load('traj_base_pos.npy')
left_angles = np.load('traj_left_angles.npy')
right_angles = np.load('traj_right_angles.npy')

print("=" * 80)
print("REALISTIC WALKING TRAJECTORIES VISUALIZATION")
print("=" * 80)

print(f"\n[TRAJECTORY DATA LOADED]")
print(f"  Duration: {times[-1]:.2f}s")
print(f"  Points: {len(times)}")
print(f"  Left foot X range: {left_foot[:, 0].min():.4f} to {left_foot[:, 0].max():.4f} m")
print(f"  Right foot X range: {right_foot[:, 0].min():.4f} to {right_foot[:, 0].max():.4f} m")

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))

# Row 1: Side view (XZ plane)
ax1 = plt.subplot(3, 3, 1)
ax1.plot(left_foot[:, 0], left_foot[:, 2], 'b-', linewidth=2.5, label='Left Foot')
ax1.plot(right_foot[:, 0], right_foot[:, 2], 'r-', linewidth=2.5, label='Right Foot')
ax1.plot(base_pos[:, 0], base_pos[:, 2], 'k--', linewidth=2, label='Base COM', alpha=0.7)
ax1.scatter(left_foot[0, 0], left_foot[0, 2], c='b', s=100, marker='o', zorder=5, label='Start')
ax1.scatter(left_foot[-1, 0], left_foot[-1, 2], c='b', s=100, marker='x', zorder=5, label='End')
ax1.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Z Position (m)', fontsize=11, fontweight='bold')
ax1.set_title('SIDE VIEW (XZ Plane)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Row 1: Top view (XY plane)
ax2 = plt.subplot(3, 3, 2)
ax2.plot(left_foot[:, 0], left_foot[:, 1], 'b-', linewidth=2, label='Left Foot')
ax2.plot(right_foot[:, 0], right_foot[:, 1], 'r-', linewidth=2, label='Right Foot')
ax2.plot(base_pos[:, 0], base_pos[:, 1], 'k--', linewidth=1.5, label='Base COM', alpha=0.7)
ax2.scatter(left_foot[0, 0], left_foot[0, 1], c='b', s=100, marker='o', zorder=5)
ax2.scatter(left_foot[-1, 0], left_foot[-1, 1], c='b', s=100, marker='x', zorder=5)
ax2.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
ax2.set_title('TOP VIEW (XY Plane)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Row 1: 3D trajectory (all 3 views)
ax3 = plt.subplot(3, 3, 3, projection='3d')
ax3.plot(left_foot[:, 0], left_foot[:, 1], left_foot[:, 2], 'b-', linewidth=2, label='Left Foot')
ax3.plot(right_foot[:, 0], right_foot[:, 1], right_foot[:, 2], 'r-', linewidth=2, label='Right Foot')
ax3.plot(base_pos[:, 0], base_pos[:, 1], base_pos[:, 2], 'k--', linewidth=1.5, label='Base', alpha=0.7)
ax3.scatter(left_foot[0, 0], left_foot[0, 1], left_foot[0, 2], c='b', s=100, marker='o')
ax3.scatter(left_foot[-1, 0], left_foot[-1, 1], left_foot[-1, 2], c='b', s=100, marker='x')
ax3.set_xlabel('X (m)', fontsize=10)
ax3.set_ylabel('Y (m)', fontsize=10)
ax3.set_zlabel('Z (m)', fontsize=10)
ax3.set_title('3D TRAJECTORIES', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)

# Row 2: X position vs time
ax4 = plt.subplot(3, 3, 4)
ax4.plot(times, left_foot[:, 0], 'b-', linewidth=2, label='Left Foot X', alpha=0.8)
ax4.plot(times, right_foot[:, 0], 'r-', linewidth=2, label='Right Foot X', alpha=0.8)
ax4.plot(times, base_pos[:, 0], 'k--', linewidth=1.5, label='Base X', alpha=0.7)
ax4.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
ax4.set_ylabel('X Position (m)', fontsize=11, fontweight='bold')
ax4.set_title('Forward Position (X) vs Time', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Row 2: Z position vs time (height)
ax5 = plt.subplot(3, 3, 5)
ax5.plot(times, left_foot[:, 2], 'b-', linewidth=2, label='Left Foot Z', alpha=0.8)
ax5.plot(times, right_foot[:, 2], 'r-', linewidth=2, label='Right Foot Z', alpha=0.8)
ax5.axhline(y=base_pos[0, 2], color='k', linestyle='--', linewidth=1.5, label='Base Height', alpha=0.7)
ax5.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Z Position (m)', fontsize=11, fontweight='bold')
ax5.set_title('Height (Z) vs Time - Swing Phases', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# Row 2: Y position vs time (lateral sway)
ax6 = plt.subplot(3, 3, 6)
ax6.plot(times, left_foot[:, 1], 'b-', linewidth=2, label='Left Foot Y', alpha=0.8)
ax6.plot(times, right_foot[:, 1], 'r-', linewidth=2, label='Right Foot Y', alpha=0.8)
ax6.plot(times, base_pos[:, 1], 'k--', linewidth=1.5, label='Base Y', alpha=0.7)
ax6.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
ax6.set_title('Lateral Motion (Y) vs Time', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# Row 3: Joint angles - Left leg
ax7 = plt.subplot(3, 3, 7)
ax7.plot(times, left_angles[:, 0], linewidth=2, label='Hip', alpha=0.8, color='#1f77b4')
ax7.plot(times, left_angles[:, 1], linewidth=2, label='Knee', alpha=0.8, color='#ff7f0e')
ax7.plot(times, left_angles[:, 2], linewidth=2, label='Ankle', alpha=0.8, color='#2ca02c')
ax7.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
ax7.set_ylabel('Angle (rad)', fontsize=11, fontweight='bold')
ax7.set_title('LEFT LEG Joint Angles', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# Row 3: Joint angles - Right leg
ax8 = plt.subplot(3, 3, 8)
ax8.plot(times, right_angles[:, 0], linewidth=2, label='Hip', alpha=0.8, color='#1f77b4')
ax8.plot(times, right_angles[:, 1], linewidth=2, label='Knee', alpha=0.8, color='#ff7f0e')
ax8.plot(times, right_angles[:, 2], linewidth=2, label='Ankle', alpha=0.8, color='#2ca02c')
ax8.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
ax8.set_ylabel('Angle (rad)', fontsize=11, fontweight='bold')
ax8.set_title('RIGHT LEG Joint Angles', fontsize=12, fontweight='bold')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)

# Row 3: Step length and swing height over time
ax9 = plt.subplot(3, 3, 9)
step_length = np.diff(left_foot[:, 0])
swing_height = np.maximum(left_foot[:, 2], right_foot[:, 2]) - 0.485  # Swing height above ground
swing_height = np.maximum(swing_height, 0)  # Clip negative values

ax9.fill_between(times, 0, swing_height, alpha=0.3, color='green', label='Swing Height')
ax9_twin = ax9.twinx()
ax9_twin.plot(times[:-1], step_length*100, 'b-', linewidth=2, label='Step Rate', alpha=0.7)

ax9.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
ax9.set_ylabel('Swing Height (m)', fontsize=11, fontweight='bold', color='green')
ax9_twin.set_ylabel('Step Rate (cm/step)', fontsize=11, fontweight='bold', color='blue')
ax9.set_title('Gait Metrics', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3)
ax9.tick_params(axis='y', labelcolor='green')
ax9_twin.tick_params(axis='y', labelcolor='blue')

plt.tight_layout()
plt.savefig('trajectory_visualization_realistic.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] Visualization saved to 'trajectory_visualization_realistic.png'")

# Summary statistics
print(f"\n[GAIT STATISTICS]")
print(f"  Stride length: {(left_foot[-1, 0] - left_foot[0, 0])*100:.1f} cm total distance")
print(f"  Walking speed: {(left_foot[-1, 0] - left_foot[0, 0])/times[-1]*100:.1f} cm/s")
print(f"  Max step height: {(left_foot[:, 2].max() - 0.485)*100:.1f} cm")
print(f"  Gait period: 2.0 s")
print(f"  Step frequency: {times[-1]/2:.1f} cycles")

print("\n" + "=" * 80)
