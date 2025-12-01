"""
Visualize the generated walking trajectories.
Plots the hip and foot positions to verify the walking gait pattern.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load trajectories
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')
base_traj = np.load('base_trajectory.npy')

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 3D view
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot(foot1_traj[:, 0], foot1_traj[:, 1], foot1_traj[:, 2], 'b-', linewidth=1.5, label='Foot 1')
ax1.plot(foot2_traj[:, 0], foot2_traj[:, 1], foot2_traj[:, 2], 'g-', linewidth=1.5, label='Foot 2')
ax1.plot(base_traj[:, 0], base_traj[:, 1], base_traj[:, 2], 'r--', linewidth=2, label='Hip')
ax1.axhline(y=0.21, color='brown', linestyle='--', linewidth=2, label='Floor')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('3D Walking Trajectory')
ax1.legend()
ax1.grid(True)

# Front view (Y-Z)
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(foot1_traj[:, 1], foot1_traj[:, 2], 'b-', linewidth=1.5, label='Foot 1')
ax2.plot(foot2_traj[:, 1], foot2_traj[:, 2], 'g-', linewidth=1.5, label='Foot 2')
ax2.plot(base_traj[:, 1], base_traj[:, 2], 'r--', linewidth=2, label='Hip')
ax2.axhline(y=0.21, color='brown', linestyle='--', linewidth=2, label='Floor')
ax2.set_xlabel('Y (m)')
ax2.set_ylabel('Z (m)')
ax2.set_title('Front View (Y-Z)')
ax2.legend()
ax2.grid(True)

# Side view (X-Z)
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(foot1_traj[:, 0], foot1_traj[:, 2], 'b-', linewidth=1.5, label='Foot 1')
ax3.plot(foot2_traj[:, 0], foot2_traj[:, 2], 'g-', linewidth=1.5, label='Foot 2')
ax3.plot(base_traj[:, 0], base_traj[:, 2], 'r--', linewidth=2, label='Hip')
ax3.axhline(y=0.21, color='brown', linestyle='--', linewidth=2, label='Floor')
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Z (m)')
ax3.set_title('Side View (X-Z)')
ax3.legend()
ax3.grid(True)

# Height over time
ax4 = fig.add_subplot(2, 3, 4)
steps = np.arange(len(foot1_traj))
ax4.plot(steps, foot1_traj[:, 2], 'b-', linewidth=1.5, label='Foot 1')
ax4.plot(steps, foot2_traj[:, 2], 'g-', linewidth=1.5, label='Foot 2')
ax4.plot(steps, base_traj[:, 2], 'r--', linewidth=2, label='Hip')
ax4.axhline(y=0.21, color='brown', linestyle='--', linewidth=2, label='Floor')
ax4.set_xlabel('Step')
ax4.set_ylabel('Height Z (m)')
ax4.set_title('Height Over Time')
ax4.legend()
ax4.grid(True)

# Forward position over time
ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(steps, foot1_traj[:, 0], 'b-', linewidth=1.5, label='Foot 1')
ax5.plot(steps, foot2_traj[:, 0], 'g-', linewidth=1.5, label='Foot 2')
ax5.plot(steps, base_traj[:, 0], 'r--', linewidth=2, label='Hip')
ax5.set_xlabel('Step')
ax5.set_ylabel('Forward Position X (m)')
ax5.set_title('Forward Progression')
ax5.legend()
ax5.grid(True)

# Lateral position over time
ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(steps, foot1_traj[:, 1], 'b-', linewidth=1.5, label='Foot 1')
ax6.plot(steps, foot2_traj[:, 1], 'g-', linewidth=1.5, label='Foot 2')
ax6.plot(steps, base_traj[:, 1], 'r--', linewidth=2, label='Hip')
ax6.set_xlabel('Step')
ax6.set_ylabel('Lateral Position Y (m)')
ax6.set_title('Lateral Separation')
ax6.legend()
ax6.grid(True)

plt.tight_layout()
plt.savefig('walking_trajectories.png', dpi=150, bbox_inches='tight')
print("[OK] Trajectory visualization saved as 'walking_trajectories.png'")
plt.show()

# Print summary
print("\n" + "="*70)
print("TRAJECTORY ANALYSIS")
print("="*70)

print(f"Foot 1 Height Range: {foot1_traj[:, 2].min():.4f}m to {foot1_traj[:, 2].max():.4f}m")
print(f"Foot 2 Height Range: {foot2_traj[:, 2].min():.4f}m to {foot2_traj[:, 2].max():.4f}m")

print(f"Floor clearance (min): {(np.minimum(foot1_traj[:, 2], foot2_traj[:, 2]).min() - 0.21)*1000:.1f}mm")

print(f"\nLateral separation: Foot 1 at Y={foot1_traj[0, 1]:.3f}m, Foot 2 at Y={foot2_traj[0, 1]:.3f}m")

print("\n" + "="*70)
