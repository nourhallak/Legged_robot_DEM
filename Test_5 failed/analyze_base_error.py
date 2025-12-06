#!/usr/bin/env python3
"""
Analyze Base (Hip) Trajectory Error

Calculates the error between planned and actual base positions.
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt

print("\n" + "="*80)
print("BASE (HIP) TRAJECTORY ERROR ANALYSIS")
print("="*80)

# Load model and data
model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
data = mujoco.MjData(model)

# Load trajectories
print("\nLoading data...")
base_planned = np.load("base_feasible.npy")
q_left = np.load("q_left_feasible.npy")
q_right = np.load("q_right_feasible.npy")

NUM_STEPS = len(q_left)
print(f"  Loaded {NUM_STEPS} trajectory points")

# Get actual base positions from forward kinematics
print("\nCalculating actual base positions...")
base_actual = np.zeros((NUM_STEPS, 3))

for step in range(NUM_STEPS):
    # Set base position and joint angles
    # Note: base position is not directly in qpos for base joints
    # We need to get it from the body position
    data.qpos[3:6] = q_left[step]
    data.qpos[6:9] = q_right[step]
    mujoco.mj_forward(model, data)
    
    # Get hip body position
    base_actual[step] = data.xpos[1].copy()  # Body 1 is the hip

print("✓ Actual positions calculated")

# Calculate errors
print("\nCalculating base trajectory errors...")
err_base = base_actual - base_planned
err_base_mag = np.linalg.norm(err_base, axis=1)

print(f"\nBase (Hip) Trajectory Error:")
print(f"  Mean magnitude: {err_base_mag.mean()*1000:.3f} mm")
print(f"  Max magnitude:  {err_base_mag.max()*1000:.3f} mm")
print(f"  Min magnitude:  {err_base_mag.min()*1000:.3f} mm")
print(f"  Std:            {err_base_mag.std()*1000:.3f} mm")

print(f"\nX-Direction Error:")
print(f"  Mean: {err_base[:, 0].mean()*1000:+.3f} mm")
print(f"  Max:  {np.abs(err_base[:, 0]).max()*1000:.3f} mm")
print(f"  Std:  {err_base[:, 0].std()*1000:.3f} mm")

print(f"\nY-Direction Error:")
print(f"  Mean: {err_base[:, 1].mean()*1000:+.3f} mm")
print(f"  Max:  {np.abs(err_base[:, 1]).max()*1000:.3f} mm")
print(f"  Std:  {err_base[:, 1].std()*1000:.3f} mm")

print(f"\nZ-Direction Error:")
print(f"  Mean: {err_base[:, 2].mean()*1000:+.3f} mm")
print(f"  Max:  {np.abs(err_base[:, 2]).max()*1000:.3f} mm")
print(f"  Std:  {err_base[:, 2].std()*1000:.3f} mm")

# Create visualization
print("\nGenerating visualizations...")
fig = plt.figure(figsize=(16, 12))

# Row 1: Overall error magnitude
ax1 = plt.subplot(3, 3, 1)
ax1.plot(err_base_mag * 1000, 'b-', linewidth=2)
ax1.fill_between(range(NUM_STEPS), 0, err_base_mag * 1000, alpha=0.3)
ax1.set_xlabel('Step')
ax1.set_ylabel('Error (mm)')
ax1.set_title('Base Position Error Magnitude')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=5, color='r', linestyle='--', label='5mm limit', alpha=0.7)
ax1.legend()

# X component error
ax2 = plt.subplot(3, 3, 2)
ax2.plot(err_base[:, 0] * 1000, 'g-', linewidth=2, label='X Error')
ax2.set_xlabel('Step')
ax2.set_ylabel('Error (mm)')
ax2.set_title('X-Direction Error')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.legend()

# Y component error
ax3 = plt.subplot(3, 3, 3)
ax3.plot(err_base[:, 1] * 1000, 'orange', linewidth=2, label='Y Error')
ax3.set_xlabel('Step')
ax3.set_ylabel('Error (mm)')
ax3.set_title('Y-Direction Error')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax3.legend()

# Z component error
ax4 = plt.subplot(3, 3, 4)
ax4.plot(err_base[:, 2] * 1000, 'purple', linewidth=2, label='Z Error')
ax4.set_xlabel('Step')
ax4.set_ylabel('Error (mm)')
ax4.set_title('Z-Direction Error')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax4.legend()

# Planned vs Actual X
ax5 = plt.subplot(3, 3, 5)
ax5.plot(base_planned[:, 0] * 1000, 'b--', label='Planned', linewidth=2, alpha=0.7)
ax5.plot(base_actual[:, 0] * 1000, 'b-', label='Actual', linewidth=2)
ax5.set_xlabel('Step')
ax5.set_ylabel('X Position (mm)')
ax5.set_title('X Trajectory: Planned vs Actual')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Planned vs Actual Y
ax6 = plt.subplot(3, 3, 6)
ax6.plot(base_planned[:, 1] * 1000, 'g--', label='Planned', linewidth=2, alpha=0.7)
ax6.plot(base_actual[:, 1] * 1000, 'g-', label='Actual', linewidth=2)
ax6.set_xlabel('Step')
ax6.set_ylabel('Y Position (mm)')
ax6.set_title('Y Trajectory: Planned vs Actual')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Planned vs Actual Z
ax7 = plt.subplot(3, 3, 7)
ax7.plot(base_planned[:, 2] * 1000, 'purple', linestyle='--', label='Planned', linewidth=2, alpha=0.7)
ax7.plot(base_actual[:, 2] * 1000, 'purple', label='Actual', linewidth=2)
ax7.set_xlabel('Step')
ax7.set_ylabel('Z Position (mm)')
ax7.set_title('Z Trajectory: Planned vs Actual')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Error distribution
ax8 = plt.subplot(3, 3, 8)
ax8.hist(err_base_mag * 1000, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax8.axvline(err_base_mag.mean() * 1000, color='r', linestyle='--', linewidth=2, label=f'Mean: {err_base_mag.mean()*1000:.2f}mm')
ax8.set_xlabel('Error (mm)')
ax8.set_ylabel('Frequency')
ax8.set_title('Error Distribution')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# 3D trajectory comparison
ax9 = plt.subplot(3, 3, 9, projection='3d')
ax9.plot(base_planned[:, 0]*1000, base_planned[:, 1]*1000, base_planned[:, 2]*1000, 
         'b--', label='Planned', linewidth=2, alpha=0.7)
ax9.plot(base_actual[:, 0]*1000, base_actual[:, 1]*1000, base_actual[:, 2]*1000, 
         'b-', label='Actual', linewidth=2)
ax9.set_xlabel('X (mm)')
ax9.set_ylabel('Y (mm)')
ax9.set_zlabel('Z (mm)')
ax9.set_title('Base 3D Trajectory')
ax9.legend()

plt.tight_layout()
plt.savefig('base_trajectory_error.png', dpi=150, bbox_inches='tight')
print("✓ Saved: base_trajectory_error.png")
plt.close()

# Create detailed report
print("\nGenerating detailed report...")
with open('base_trajectory_error_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("BASE (HIP) TRAJECTORY ERROR ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Total Steps Analyzed: {NUM_STEPS}\n\n")
    
    f.write("OVERALL ERROR STATISTICS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Mean Error: {err_base_mag.mean()*1000:.3f} mm\n")
    f.write(f"Max Error:  {err_base_mag.max()*1000:.3f} mm\n")
    f.write(f"Min Error:  {err_base_mag.min()*1000:.3f} mm\n")
    f.write(f"Std Dev:    {err_base_mag.std()*1000:.3f} mm\n\n")
    
    f.write("COMPONENT-WISE ERRORS (mm)\n")
    f.write("-" * 40 + "\n")
    f.write(f"X: Mean {err_base[:, 0].mean()*1000:+.3f}, Max {np.abs(err_base[:, 0]).max()*1000:.3f}, Std {err_base[:, 0].std()*1000:.3f}\n")
    f.write(f"Y: Mean {err_base[:, 1].mean()*1000:+.3f}, Max {np.abs(err_base[:, 1]).max()*1000:.3f}, Std {err_base[:, 1].std()*1000:.3f}\n")
    f.write(f"Z: Mean {err_base[:, 2].mean()*1000:+.3f}, Max {np.abs(err_base[:, 2]).max()*1000:.3f}, Std {err_base[:, 2].std()*1000:.3f}\n\n")
    
    f.write("TRAJECTORY RANGES (mm)\n")
    f.write("-" * 40 + "\n")
    f.write("PLANNED:\n")
    f.write(f"  X: {base_planned[:, 0].min()*1000:.2f} to {base_planned[:, 0].max()*1000:.2f}\n")
    f.write(f"  Y: {base_planned[:, 1].min()*1000:.2f} to {base_planned[:, 1].max()*1000:.2f}\n")
    f.write(f"  Z: {base_planned[:, 2].min()*1000:.2f} to {base_planned[:, 2].max()*1000:.2f}\n\n")
    
    f.write("ACTUAL:\n")
    f.write(f"  X: {base_actual[:, 0].min()*1000:.2f} to {base_actual[:, 0].max()*1000:.2f}\n")
    f.write(f"  Y: {base_actual[:, 1].min()*1000:.2f} to {base_actual[:, 1].max()*1000:.2f}\n")
    f.write(f"  Z: {base_actual[:, 2].min()*1000:.2f} to {base_actual[:, 2].max()*1000:.2f}\n\n")
    
    f.write("ERROR ACCEPTABILITY\n")
    f.write("-" * 40 + "\n")
    f.write(f"Error < 5mm: {np.sum(err_base_mag < 0.005)}/{NUM_STEPS} ({np.sum(err_base_mag < 0.005)/NUM_STEPS*100:.1f}%)\n")
    f.write(f"Error < 2mm: {np.sum(err_base_mag < 0.002)}/{NUM_STEPS} ({np.sum(err_base_mag < 0.002)/NUM_STEPS*100:.1f}%)\n")
    f.write(f"Error < 1mm: {np.sum(err_base_mag < 0.001)}/{NUM_STEPS} ({np.sum(err_base_mag < 0.001)/NUM_STEPS*100:.1f}%)\n\n")
    
    f.write("WORST CASES (Top 5 errors)\n")
    f.write("-" * 40 + "\n")
    worst_indices = np.argsort(err_base_mag)[-5:][::-1]
    for i, idx in enumerate(worst_indices, 1):
        f.write(f"{i}. Step {idx}: {err_base_mag[idx]*1000:.3f}mm ")
        f.write(f"[X:{err_base[idx, 0]*1000:+.2f}, Y:{err_base[idx, 1]*1000:+.2f}, Z:{err_base[idx, 2]*1000:+.2f}]\n")

print("✓ Saved: base_trajectory_error_report.txt")

# Save error data
np.save('err_base.npy', err_base)
np.save('base_actual.npy', base_actual)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nSummary:")
print(f"  Mean base error: {err_base_mag.mean()*1000:.3f} mm")
print(f"  Max base error:  {err_base_mag.max()*1000:.3f} mm")
print(f"  Points < 5mm:    {np.sum(err_base_mag < 0.005)}/{NUM_STEPS} ({np.sum(err_base_mag < 0.005)/NUM_STEPS*100:.1f}%)")
print("\nFiles generated:")
print("  - base_trajectory_error.png")
print("  - base_trajectory_error_report.txt")
print("  - err_base.npy, base_actual.npy\n")
