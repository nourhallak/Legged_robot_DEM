#!/usr/bin/env python3
"""
Analyze IK Error: Compare Planned vs Actual Trajectories

Calculates the error between planned foot trajectories and 
actual foot positions achieved by IK solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
import mujoco

print("\n" + "="*80)
print("INVERSE KINEMATICS ERROR ANALYSIS")
print("="*80)

# Load model and data
model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
data = mujoco.MjData(model)

# Load trajectories
print("\nLoading data...")
base_planned = np.load("base_feasible.npy")
foot1_planned = np.load("foot1_feasible.npy")
foot2_planned = np.load("foot2_feasible.npy")
q_left = np.load("q_left_feasible.npy")
q_right = np.load("q_right_feasible.npy")

NUM_STEPS = len(q_left)
print(f"  Loaded {NUM_STEPS} trajectory points")

# Calculate actual foot positions from IK solutions
print("\nCalculating actual foot positions from joint angles...")
foot1_actual = np.zeros((NUM_STEPS, 3))
foot2_actual = np.zeros((NUM_STEPS, 3))

for step in range(NUM_STEPS):
    # Set joint angles from IK solutions
    data.qpos[3:6] = q_left[step]
    data.qpos[6:9] = q_right[step]
    
    # Forward kinematics
    mujoco.mj_forward(model, data)
    
    # Get actual foot positions
    foot1_actual[step] = data.site_xpos[0].copy()
    foot2_actual[step] = data.site_xpos[1].copy()

print("✓ Actual positions calculated")

# Calculate errors
print("\nCalculating errors...")
err_foot1 = foot1_actual - foot1_planned
err_foot2 = foot2_actual - foot2_planned

err_foot1_mag = np.linalg.norm(err_foot1, axis=1)
err_foot2_mag = np.linalg.norm(err_foot2, axis=1)

print(f"\nLeft Foot (Foot1) Error:")
print(f"  Mean: {err_foot1_mag.mean()*1000:.3f} mm")
print(f"  Max:  {err_foot1_mag.max()*1000:.3f} mm")
print(f"  Min:  {err_foot1_mag.min()*1000:.3f} mm")
print(f"  Std:  {err_foot1_mag.std()*1000:.3f} mm")

print(f"\nRight Foot (Foot2) Error:")
print(f"  Mean: {err_foot2_mag.mean()*1000:.3f} mm")
print(f"  Max:  {err_foot2_mag.max()*1000:.3f} mm")
print(f"  Min:  {err_foot2_mag.min()*1000:.3f} mm")
print(f"  Std:  {err_foot2_mag.std()*1000:.3f} mm")

# Component-wise errors
print(f"\nLeft Foot Component Errors (mm):")
print(f"  X: mean={err_foot1[:, 0].mean()*1000:.3f}, max={np.abs(err_foot1[:, 0]).max()*1000:.3f}")
print(f"  Y: mean={err_foot1[:, 1].mean()*1000:.3f}, max={np.abs(err_foot1[:, 1]).max()*1000:.3f}")
print(f"  Z: mean={err_foot1[:, 2].mean()*1000:.3f}, max={np.abs(err_foot1[:, 2]).max()*1000:.3f}")

print(f"\nRight Foot Component Errors (mm):")
print(f"  X: mean={err_foot2[:, 0].mean()*1000:.3f}, max={np.abs(err_foot2[:, 0]).max()*1000:.3f}")
print(f"  Y: mean={err_foot2[:, 1].mean()*1000:.3f}, max={np.abs(err_foot2[:, 1]).max()*1000:.3f}")
print(f"  Z: mean={err_foot2[:, 2].mean()*1000:.3f}, max={np.abs(err_foot2[:, 2]).max()*1000:.3f}")

# Create visualization
print("\nGenerating visualization...")
fig = plt.figure(figsize=(15, 12))

# Row 1: Overall error magnitude
ax1 = plt.subplot(3, 3, 1)
ax1.plot(err_foot1_mag * 1000, 'b-', label='Left Foot', alpha=0.7)
ax1.plot(err_foot2_mag * 1000, 'r-', label='Right Foot', alpha=0.7)
ax1.set_xlabel('Step')
ax1.set_ylabel('Error (mm)')
ax1.set_title('IK Error Magnitude')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Row 1: X component
ax2 = plt.subplot(3, 3, 2)
ax2.plot(err_foot1[:, 0] * 1000, 'b-', label='Left Foot', alpha=0.7)
ax2.plot(err_foot2[:, 0] * 1000, 'r-', label='Right Foot', alpha=0.7)
ax2.set_xlabel('Step')
ax2.set_ylabel('Error (mm)')
ax2.set_title('X-Direction Error')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Row 1: Y component
ax3 = plt.subplot(3, 3, 3)
ax3.plot(err_foot1[:, 1] * 1000, 'b-', label='Left Foot', alpha=0.7)
ax3.plot(err_foot2[:, 1] * 1000, 'r-', label='Right Foot', alpha=0.7)
ax3.set_xlabel('Step')
ax3.set_ylabel('Error (mm)')
ax3.set_title('Y-Direction Error')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Row 2: Z component
ax4 = plt.subplot(3, 3, 4)
ax4.plot(err_foot1[:, 2] * 1000, 'b-', label='Left Foot', alpha=0.7)
ax4.plot(err_foot2[:, 2] * 1000, 'r-', label='Right Foot', alpha=0.7)
ax4.set_xlabel('Step')
ax4.set_ylabel('Error (mm)')
ax4.set_title('Z-Direction Error')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Row 2: Planned vs Actual X
ax5 = plt.subplot(3, 3, 5)
ax5.plot(foot1_planned[:, 0] * 1000, 'b--', label='Left Planned', alpha=0.7)
ax5.plot(foot1_actual[:, 0] * 1000, 'b-', label='Left Actual', alpha=0.7)
ax5.plot(foot2_planned[:, 0] * 1000, 'r--', label='Right Planned', alpha=0.7)
ax5.plot(foot2_actual[:, 0] * 1000, 'r-', label='Right Actual', alpha=0.7)
ax5.set_xlabel('Step')
ax5.set_ylabel('X Position (mm)')
ax5.set_title('X Trajectory: Planned vs Actual')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Row 2: Planned vs Actual Y
ax6 = plt.subplot(3, 3, 6)
ax6.plot(foot1_planned[:, 1] * 1000, 'b--', label='Left Planned', alpha=0.7)
ax6.plot(foot1_actual[:, 1] * 1000, 'b-', label='Left Actual', alpha=0.7)
ax6.plot(foot2_planned[:, 1] * 1000, 'r--', label='Right Planned', alpha=0.7)
ax6.plot(foot2_actual[:, 1] * 1000, 'r-', label='Right Actual', alpha=0.7)
ax6.set_xlabel('Step')
ax6.set_ylabel('Y Position (mm)')
ax6.set_title('Y Trajectory: Planned vs Actual')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Row 3: Planned vs Actual Z
ax7 = plt.subplot(3, 3, 7)
ax7.plot(foot1_planned[:, 2] * 1000, 'b--', label='Left Planned', alpha=0.7)
ax7.plot(foot1_actual[:, 2] * 1000, 'b-', label='Left Actual', alpha=0.7)
ax7.plot(foot2_planned[:, 2] * 1000, 'r--', label='Right Planned', alpha=0.7)
ax7.plot(foot2_actual[:, 2] * 1000, 'r-', label='Right Actual', alpha=0.7)
ax7.set_xlabel('Step')
ax7.set_ylabel('Z Position (mm)')
ax7.set_title('Z Trajectory: Planned vs Actual')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# Row 3: Error distribution
ax8 = plt.subplot(3, 3, 8)
ax8.hist(err_foot1_mag * 1000, bins=30, alpha=0.7, label='Left Foot', color='blue')
ax8.hist(err_foot2_mag * 1000, bins=30, alpha=0.7, label='Right Foot', color='red')
ax8.set_xlabel('Error (mm)')
ax8.set_ylabel('Frequency')
ax8.set_title('Error Distribution')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Row 3: 3D error vectors at key points
ax9 = plt.subplot(3, 3, 9, projection='3d')
steps_to_show = [0, 50, 100, 150, 200, 250, 299]
for i, step in enumerate(steps_to_show):
    color = plt.cm.viridis(i / len(steps_to_show))
    # Left foot error
    ax9.quiver(foot1_planned[step, 0]*1000, foot1_planned[step, 1]*1000, foot1_planned[step, 2]*1000,
              err_foot1[step, 0]*1000, err_foot1[step, 1]*1000, err_foot1[step, 2]*1000,
              color=color, alpha=0.7, arrow_length_ratio=0.3)

ax9.set_xlabel('X (mm)')
ax9.set_ylabel('Y (mm)')
ax9.set_zlabel('Z (mm)')
ax9.set_title('Error Vectors (Left Foot)')

plt.tight_layout()
plt.savefig('ik_error_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: ik_error_analysis.png")
plt.close()

# Create summary report
print("\nGenerating error summary report...")
with open('ik_error_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("INVERSE KINEMATICS ERROR ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Total Steps Analyzed: {NUM_STEPS}\n\n")
    
    f.write("LEFT FOOT ERROR STATISTICS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total Magnitude Error:\n")
    f.write(f"  Mean: {err_foot1_mag.mean()*1000:.3f} mm\n")
    f.write(f"  Std:  {err_foot1_mag.std()*1000:.3f} mm\n")
    f.write(f"  Min:  {err_foot1_mag.min()*1000:.3f} mm\n")
    f.write(f"  Max:  {err_foot1_mag.max()*1000:.3f} mm\n\n")
    
    f.write("Component Errors (mm):\n")
    f.write(f"  X: {err_foot1[:, 0].mean()*1000:+.3f} ± {err_foot1[:, 0].std()*1000:.3f}\n")
    f.write(f"  Y: {err_foot1[:, 1].mean()*1000:+.3f} ± {err_foot1[:, 1].std()*1000:.3f}\n")
    f.write(f"  Z: {err_foot1[:, 2].mean()*1000:+.3f} ± {err_foot1[:, 2].std()*1000:.3f}\n\n")
    
    f.write("RIGHT FOOT ERROR STATISTICS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total Magnitude Error:\n")
    f.write(f"  Mean: {err_foot2_mag.mean()*1000:.3f} mm\n")
    f.write(f"  Std:  {err_foot2_mag.std()*1000:.3f} mm\n")
    f.write(f"  Min:  {err_foot2_mag.min()*1000:.3f} mm\n")
    f.write(f"  Max:  {err_foot2_mag.max()*1000:.3f} mm\n\n")
    
    f.write("Component Errors (mm):\n")
    f.write(f"  X: {err_foot2[:, 0].mean()*1000:+.3f} ± {err_foot2[:, 0].std()*1000:.3f}\n")
    f.write(f"  Y: {err_foot2[:, 1].mean()*1000:+.3f} ± {err_foot2[:, 1].std()*1000:.3f}\n")
    f.write(f"  Z: {err_foot2[:, 2].mean()*1000:+.3f} ± {err_foot2[:, 2].std()*1000:.3f}\n\n")
    
    f.write("ERROR ACCEPTABILITY\n")
    f.write("-" * 40 + "\n")
    f.write(f"Left foot < 5mm: {np.sum(err_foot1_mag < 0.005)}/{NUM_STEPS} ({np.sum(err_foot1_mag < 0.005)/NUM_STEPS*100:.1f}%)\n")
    f.write(f"Right foot < 5mm: {np.sum(err_foot2_mag < 0.005)}/{NUM_STEPS} ({np.sum(err_foot2_mag < 0.005)/NUM_STEPS*100:.1f}%)\n")
    f.write(f"Both < 5mm: {np.sum((err_foot1_mag < 0.005) & (err_foot2_mag < 0.005))}/{NUM_STEPS}\n")

print("✓ Saved: ik_error_report.txt")

# Save error data
np.save('err_foot1_actual.npy', err_foot1)
np.save('err_foot2_actual.npy', err_foot2)
np.save('foot1_actual.npy', foot1_actual)
np.save('foot2_actual.npy', foot2_actual)

print("\n" + "="*80)
print("ERROR ANALYSIS COMPLETE")
print("="*80)
print("\nSummary:")
print(f"  Left foot mean error:  {err_foot1_mag.mean()*1000:.3f} mm")
print(f"  Right foot mean error: {err_foot2_mag.mean()*1000:.3f} mm")
print("\nFiles generated:")
print("  - ik_error_analysis.png")
print("  - ik_error_report.txt")
print("  - err_foot1_actual.npy, err_foot2_actual.npy")
print("  - foot1_actual.npy, foot2_actual.npy\n")
