#!/usr/bin/env python3
"""
Comprehensive knee angle analysis to determine correct joint limits.
Tests how foot Z-height varies with knee angle to find realistic walking range.
"""
import numpy as np
import mujoco
import os

# Load the model
script_dir = os.path.dirname(os.path.abspath(__file__))
mjcf_path = os.path.join(script_dir, "legged_robot_ik.xml")

try:
    model = mujoco.MjModel.from_xml_path(mjcf_path)
except:
    model = mujoco.load_model_from_xml(open(mjcf_path).read())

data = mujoco.MjData(model)

print("=" * 80)
print("KNEE ANGLE ANALYSIS - Finding optimal joint limits")
print("=" * 80)

# Test parameters: hip and ankle fixed, sweep knee
hip_angle = 0.0  # Hip joint neutral
ankle_angle = 0.0  # Ankle joint neutral

print(f"\nFixed joint angles:")
print(f"  Hip: {np.degrees(hip_angle):.1f} deg")
print(f"  Ankle: {np.degrees(ankle_angle):.1f} deg")

# Sweep knee from -180 to +180 degrees
knee_angles_deg = np.linspace(-180, 180, 361)
knee_angles_rad = np.radians(knee_angles_deg)

results = []

for knee_angle in knee_angles_rad:
    # Set floating base
    data.qpos[0:6] = [0, 0, 0.2, 1, 0, 0]  # Position: (0, 0, 0.2), Quat: (w=1, xyz=0)
    
    # Left leg (indices 6, 7, 8)
    data.qpos[6] = hip_angle
    data.qpos[7] = knee_angle
    data.qpos[8] = ankle_angle
    
    # Right leg (indices 9, 10, 11)
    data.qpos[9] = hip_angle
    data.qpos[10] = knee_angle
    data.qpos[11] = ankle_angle
    
    # Compute forward kinematics
    mujoco.mj_kinematics(model, data)
    
    # Get foot positions
    foot1_pos = data.site_xpos[1]  # site_1 = foot1
    foot2_pos = data.site_xpos[2]  # site_2 = foot2
    
    results.append({
        'knee_deg': np.degrees(knee_angle),
        'knee_rad': knee_angle,
        'foot1_x': foot1_pos[0],
        'foot1_z': foot1_pos[2],
        'foot2_x': foot2_pos[0],
        'foot2_z': foot2_pos[2],
    })

# Convert to arrays for easier access
knee_deg_arr = np.array([r['knee_deg'] for r in results])
foot1_z_arr = np.array([r['foot1_z'] for r in results])
foot2_z_arr = np.array([r['foot2_z'] for r in results])

# Find useful ranges
print("\n" + "=" * 80)
print("FULL SWEEP RESULTS")
print("=" * 80)

print(f"\nFoot 1 (Left leg) Z range: {foot1_z_arr.min():.6f} to {foot1_z_arr.max():.6f} m")
print(f"Foot 2 (Right leg) Z range: {foot2_z_arr.min():.6f} to {foot2_z_arr.max():.6f} m")

# Find where feet are at similar height (good for walking)
min_z = max(foot1_z_arr.min(), foot2_z_arr.min())
max_z = min(foot1_z_arr.max(), foot2_z_arr.max())
print(f"\nCommon Z range (both feet can reach): {min_z:.6f} to {max_z:.6f} m")

# Display detailed table for key angles
print("\n" + "=" * 80)
print("KEY ANGLES - Detailed Analysis")
print("=" * 80)
print(f"{'Knee (deg)':>12} | {'Foot1 X':>8} | {'Foot1 Z':>8} | {'Foot2 X':>8} | {'Foot2 Z':>8}")
print("-" * 70)

sample_angles = [-180, -150, -120, -90, -60, -45, -30, 0, 30, 45, 60, 90, 120, 150, 180]
for angle_deg in sample_angles:
    idx = np.argmin(np.abs(knee_deg_arr - angle_deg))
    r = results[idx]
    print(f"{r['knee_deg']:>12.1f} | {r['foot1_x']:>8.5f} | {r['foot1_z']:>8.5f} | {r['foot2_x']:>8.5f} | {r['foot2_z']:>8.5f}")

# Find angles where feet are at minimum height (good for stance)
print("\n" + "=" * 80)
print("STANCE PHASE ANALYSIS")
print("=" * 80)

# Find minimum Z configurations
stance_threshold = min_z + 0.002  # Within 2mm of minimum
stance_indices = np.where(foot1_z_arr <= stance_threshold)[0]
if len(stance_indices) > 0:
    stance_angles_deg = knee_deg_arr[stance_indices]
    print(f"\nAngles where foot1 is at minimum Z ({min_z:.6f}m, +2mm tolerance):")
    print(f"  Range: {stance_angles_deg.min():.1f}° to {stance_angles_deg.max():.1f}°")
    print(f"  Span: {stance_angles_deg.max() - stance_angles_deg.min():.1f}°")

# Find max foot separation for swing
print("\n" + "=" * 80)
print("SWING PHASE ANALYSIS")
print("=" * 80)

# Find maximum Z (highest swing possible)
max_z_idx = np.argmax(foot1_z_arr)
print(f"\nMaximum Z reachable: {foot1_z_arr.max():.6f} m")
print(f"  Achieved at knee angle: {knee_deg_arr[max_z_idx]:.1f}°")

z_range = foot1_z_arr.max() - foot1_z_arr.min()
print(f"\nTotal Z range available: {z_range:.6f} m ({z_range*1000:.2f} mm)")
print(f"Recommended swing clearance: {z_range * 0.7:.6f} m ({z_range * 0.7 * 1000:.2f} mm)")

# Recommend limits
print("\n" + "=" * 80)
print("RECOMMENDED KNEE JOINT LIMITS")
print("=" * 80)

# Strategy: use angles that give good stance (low Z) and good swing range
# Find the angle with minimum Z
min_z_idx = np.argmin(foot1_z_arr)
best_stance_angle = knee_deg_arr[min_z_idx]

# Find range that allows 80% of max Z (good swing clearance)
swing_threshold = foot1_z_arr.min() + (z_range * 0.8)
good_swing_indices = np.where(foot1_z_arr >= swing_threshold)[0]

if len(good_swing_indices) > 0:
    swing_angles = knee_deg_arr[good_swing_indices]
    swing_min = swing_angles.min()
    swing_max = swing_angles.max()
    print(f"\nAngles allowing 80% of max swing height:")
    print(f"  Range: {swing_min:.1f}° to {swing_max:.1f}°")
else:
    swing_min, swing_max = -90, 90
    print(f"\nUsing default swing range: {swing_min:.1f}° to {swing_max:.1f}°")

# Estimate good walking range
print(f"\nFor natural walking:")
print(f"  Stance (low Z): around {best_stance_angle:.1f}°")
print(f"  Swing range: {swing_min:.1f}° to {swing_max:.1f}°")

# Conservative recommendations
lower_limit = min(best_stance_angle - 30, -120)  # Some margin
upper_limit = max(best_stance_angle + 30, 60)    # Some margin

print(f"\nConservative limits (with margin): {lower_limit:.0f}° to {upper_limit:.0f}°")
print(f"  In radians: {np.radians(lower_limit):.4f} to {np.radians(upper_limit):.4f}")

# Export recommendation
recommendation = {
    'lower_deg': lower_limit,
    'upper_deg': upper_limit,
    'lower_rad': np.radians(lower_limit),
    'upper_rad': np.radians(upper_limit),
    'stance_angle': best_stance_angle,
    'z_min': foot1_z_arr.min(),
    'z_max': foot1_z_arr.max(),
    'z_range': z_range,
}

print("\n" + "=" * 80)
print(f"\nFinal Recommendation:")
print(f"  Lower limit: {recommendation['lower_deg']:.1f}° ({recommendation['lower_rad']:.4f} rad)")
print(f"  Upper limit: {recommendation['upper_deg']:.1f}° ({recommendation['upper_rad']:.4f} rad)")
print(f"  Best stance angle: {recommendation['stance_angle']:.1f}°")
print(f"  Available Z range: {recommendation['z_min']:.6f} to {recommendation['z_max']:.6f} m")
