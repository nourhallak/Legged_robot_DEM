"""
Plot and save walking trajectories as PNG
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load trajectories
hip_traj = np.load('hip_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

steps = np.arange(len(hip_traj))

print("Loading trajectories for plotting...")
print(f"Hip trajectory shape: {hip_traj.shape}")
print(f"Foot1 trajectory shape: {foot1_traj.shape}")
print(f"Foot2 trajectory shape: {foot2_traj.shape}")

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Bipedal Walking Trajectories', fontsize=16, fontweight='bold')

# ===== Plot 1: Z (Height) Trajectories =====
ax = axes[0]
ax.plot(steps, hip_traj[:, 2], 'b-', linewidth=2.5, label='Hip (Base)', marker='o', markersize=2, markevery=50)
ax.plot(steps, foot1_traj[:, 2], 'r--', linewidth=2, label='Foot 1 (Left)', marker='s', markersize=2, markevery=50)
ax.plot(steps, foot2_traj[:, 2], 'g--', linewidth=2, label='Foot 2 (Right)', marker='^', markersize=2, markevery=50)
ax.axhline(y=0.21, color='k', linestyle=':', linewidth=1.5, label='Floor (z=0.21m)', alpha=0.7)
ax.set_ylabel('Height Z (m)', fontsize=12, fontweight='bold')
ax.set_title('Vertical Position (Height)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11)
ax.set_xlim([0, len(steps)-1])

# Add annotations for key heights
ax.text(200, hip_traj[:, 2].max() + 0.003, f'Hip Max: {hip_traj[:, 2].max():.4f}m', 
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(200, hip_traj[:, 2].min() - 0.004, f'Hip Min: {hip_traj[:, 2].min():.4f}m', 
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# ===== Plot 2: X (Forward) Trajectories =====
ax = axes[1]
ax.plot(steps, hip_traj[:, 0], 'b-', linewidth=2.5, label='Hip (Base)', marker='o', markersize=2, markevery=50)
ax.plot(steps, foot1_traj[:, 0], 'r--', linewidth=2, label='Foot 1 (Left)', marker='s', markersize=2, markevery=50)
ax.plot(steps, foot2_traj[:, 0], 'g--', linewidth=2, label='Foot 2 (Right)', marker='^', markersize=2, markevery=50)
ax.set_ylabel('Forward Position X (m)', fontsize=12, fontweight='bold')
ax.set_title('Forward Progress (X Direction)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11)
ax.set_xlim([0, len(steps)-1])

# ===== Plot 3: Y (Lateral) Trajectories =====
ax = axes[2]
ax.plot(steps, hip_traj[:, 1], 'b-', linewidth=2.5, label='Hip (Base)', marker='o', markersize=2, markevery=50)
ax.plot(steps, foot1_traj[:, 1], 'r--', linewidth=2, label='Foot 1 (Left)', marker='s', markersize=2, markevery=50)
ax.plot(steps, foot2_traj[:, 1], 'g--', linewidth=2, label='Foot 2 (Right)', marker='^', markersize=2, markevery=50)
ax.set_xlabel('Gait Cycle Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Lateral Position Y (m)', fontsize=12, fontweight='bold')
ax.set_title('Lateral Separation (Y Direction)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11)
ax.set_xlim([0, len(steps)-1])

plt.tight_layout()
plt.savefig('trajectories.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Trajectory plot saved as 'trajectories.png' (300 DPI)")

# Create second figure: Side view (X-Z plane) - Physical visualization
fig, ax = plt.subplots(figsize=(14, 8))

# Plot trajectory lines
ax.plot(hip_traj[:, 0], hip_traj[:, 2], 'b-', linewidth=3, label='Hip Path', alpha=0.8)
ax.plot(foot1_traj[:, 0], foot1_traj[:, 2], 'r--', linewidth=2, label='Foot 1 Path', alpha=0.7)
ax.plot(foot2_traj[:, 0], foot2_traj[:, 2], 'g--', linewidth=2, label='Foot 2 Path', alpha=0.7)

# Plot key frames (every 50 steps)
for i in range(0, len(hip_traj), 50):
    # Hip position
    ax.plot(hip_traj[i, 0], hip_traj[i, 2], 'bo', markersize=8, alpha=0.6)
    
    # Feet positions
    ax.plot(foot1_traj[i, 0], foot1_traj[i, 2], 'rs', markersize=6, alpha=0.5)
    ax.plot(foot2_traj[i, 0], foot2_traj[i, 2], 'g^', markersize=6, alpha=0.5)
    
    # Draw legs connecting hip to feet
    ax.plot([hip_traj[i, 0], foot1_traj[i, 0]], [hip_traj[i, 2], foot1_traj[i, 2]], 
            'r-', alpha=0.3, linewidth=1)
    ax.plot([hip_traj[i, 0], foot2_traj[i, 0]], [hip_traj[i, 2], foot2_traj[i, 2]], 
            'g-', alpha=0.3, linewidth=1)

# Floor
ax.axhline(y=0.21, color='brown', linestyle='-', linewidth=4, label='Floor', alpha=0.8)
ax.fill_between(ax.get_xlim(), -0.05, 0.21, color='brown', alpha=0.1)

ax.set_xlabel('Forward Position X (m)', fontsize=13, fontweight='bold')
ax.set_ylabel('Height Z (m)', fontsize=13, fontweight='bold')
ax.set_title('Side View of Walking Motion (X-Z Plane)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=12)
ax.set_aspect('equal', adjustable='box')
ax.set_ylim([0.15, 0.35])

plt.tight_layout()
plt.savefig('walking_sideview.png', dpi=300, bbox_inches='tight')
print(f"✓ Side view plot saved as 'walking_sideview.png' (300 DPI)")

# Create third figure: 3D visualization
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectories in 3D
ax.plot(hip_traj[:, 0], hip_traj[:, 1], hip_traj[:, 2], 'b-', linewidth=3, label='Hip', alpha=0.8)
ax.plot(foot1_traj[:, 0], foot1_traj[:, 1], foot1_traj[:, 2], 'r--', linewidth=2, label='Foot 1', alpha=0.7)
ax.plot(foot2_traj[:, 0], foot2_traj[:, 1], foot2_traj[:, 2], 'g--', linewidth=2, label='Foot 2', alpha=0.7)

# Plot key frames
for i in range(0, len(hip_traj), 100):
    ax.scatter(hip_traj[i, 0], hip_traj[i, 1], hip_traj[i, 2], c='blue', s=100, marker='o', alpha=0.7)
    ax.scatter(foot1_traj[i, 0], foot1_traj[i, 1], foot1_traj[i, 2], c='red', s=50, marker='s', alpha=0.6)
    ax.scatter(foot2_traj[i, 0], foot2_traj[i, 1], foot2_traj[i, 2], c='green', s=50, marker='^', alpha=0.6)

# Add floor plane
xx, yy = np.meshgrid(np.linspace(hip_traj[:, 0].min()-0.01, hip_traj[:, 0].max()+0.01, 10),
                      np.linspace(-0.05, 0.05, 10))
zz = np.ones_like(xx) * 0.21
ax.plot_surface(xx, yy, zz, alpha=0.2, color='brown')

ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
ax.set_title('3D Walking Trajectory', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)

plt.savefig('walking_3d.png', dpi=300, bbox_inches='tight')
print(f"✓ 3D view plot saved as 'walking_3d.png' (300 DPI)")

print("\n" + "="*70)
print("TRAJECTORY ANALYSIS SUMMARY")
print("="*70)
print(f"\nHip Trajectory:")
print(f"  X range: {hip_traj[:, 0].min():.4f}m to {hip_traj[:, 0].max():.4f}m (stride: {hip_traj[:, 0].max()-hip_traj[:, 0].min():.4f}m)")
print(f"  Y range: {hip_traj[:, 1].min():.4f}m to {hip_traj[:, 1].max():.4f}m")
print(f"  Z range: {hip_traj[:, 2].min():.4f}m to {hip_traj[:, 2].max():.4f}m (oscillation: {hip_traj[:, 2].max()-hip_traj[:, 2].min():.4f}m)")

print(f"\nFoot 1 Trajectory:")
print(f"  X range: {foot1_traj[:, 0].min():.4f}m to {foot1_traj[:, 0].max():.4f}m")
print(f"  Y range: {foot1_traj[:, 1].min():.4f}m to {foot1_traj[:, 1].max():.4f}m")
print(f"  Z range: {foot1_traj[:, 2].min():.4f}m to {foot1_traj[:, 2].max():.4f}m")

print(f"\nFoot 2 Trajectory:")
print(f"  X range: {foot2_traj[:, 0].min():.4f}m to {foot2_traj[:, 0].max():.4f}m")
print(f"  Y range: {foot2_traj[:, 1].min():.4f}m to {foot2_traj[:, 1].max():.4f}m")
print(f"  Z range: {foot2_traj[:, 2].min():.4f}m to {foot2_traj[:, 2].max():.4f}m")

print(f"\nGround Clearance:")
print(f"  Floor level: 0.2100m")
print(f"  Min foot height: {min(foot1_traj[:, 2].min(), foot2_traj[:, 2].min()):.4f}m")
print(f"  Clearance: {min(foot1_traj[:, 2].min(), foot2_traj[:, 2].min()) - 0.21:.4f}m")

print(f"\nPhysical Relationships:")
print(f"  Hip below foot maximum: {foot1_traj[:, 2].max() - hip_traj[:, 2].min():.4f}m")
print(f"  Hip above foot minimum: {hip_traj[:, 2].min() - foot1_traj[:, 2].min():.4f}m")

print("\n" + "="*70)
print("All plots saved successfully!")
print("="*70 + "\n")
