#!/usr/bin/env python3
"""
Diagnose the walking simulation - check joint angles and motion
"""

import numpy as np
import mujoco
from scipy.interpolate import interp1d

# Load model
model = mujoco.MjModel.from_xml_path("legged_robot_sand_low_damping.xml")
data = mujoco.MjData(model)

# Load IK
ik_times = np.load("ik_times.npy")
ik_left_hip = np.load("ik_left_hip.npy")
ik_left_knee = np.load("ik_left_knee.npy")
ik_left_ankle = np.load("ik_left_ankle.npy")
ik_right_hip = np.load("ik_right_hip.npy")
ik_right_knee = np.load("ik_right_knee.npy")
ik_right_ankle = np.load("ik_right_ankle.npy")

print("="*70)
print("GAIT DIAGNOSTIC")
print("="*70)
print(f"\nIK Trajectory Data:")
print(f"  Time range: {ik_times[0]:.2f} to {ik_times[-1]:.2f}s")
print(f"  Total period: {ik_times[-1] - ik_times[0]:.2f}s")
print(f"  Number of points: {len(ik_times)}")

print(f"\nLeft Hip angles:")
print(f"  Min: {np.min(ik_left_hip):+.4f} rad ({np.min(ik_left_hip)*180/np.pi:+.1f}°)")
print(f"  Max: {np.max(ik_left_hip):+.4f} rad ({np.max(ik_left_hip)*180/np.pi:+.1f}°)")
print(f"  Range: {np.max(ik_left_hip) - np.min(ik_left_hip):.4f} rad")

print(f"\nRight Hip angles:")
print(f"  Min: {np.min(ik_right_hip):+.4f} rad ({np.min(ik_right_hip)*180/np.pi:+.1f}°)")
print(f"  Max: {np.max(ik_right_hip):+.4f} rad ({np.max(ik_right_hip)*180/np.pi:+.1f}°)")
print(f"  Range: {np.max(ik_right_hip) - np.min(ik_right_hip):.4f} rad")

print(f"\nLeft Knee angles:")
print(f"  Min: {np.min(ik_left_knee):+.4f} rad ({np.min(ik_left_knee)*180/np.pi:+.1f}°)")
print(f"  Max: {np.max(ik_left_knee):+.4f} rad ({np.max(ik_left_knee)*180/np.pi:+.1f}°)")
print(f"  Range: {np.max(ik_left_knee) - np.min(ik_left_knee):.4f} rad")

print(f"\nRight Knee angles:")
print(f"  Min: {np.min(ik_right_knee):+.4f} rad ({np.min(ik_right_knee)*180/np.pi:+.1f}°)")
print(f"  Max: {np.max(ik_right_knee):+.4f} rad ({np.max(ik_right_knee)*180/np.pi:+.1f}°)")
print(f"  Range: {np.max(ik_right_knee) - np.min(ik_right_knee):.4f} rad")

# Check if there's alternation
lh_min_idx = np.argmin(ik_left_hip)
rh_min_idx = np.argmin(ik_right_hip)

print(f"\nAlternation Check:")
print(f"  Left hip minimum at t={ik_times[lh_min_idx]:.2f}s")
print(f"  Right hip minimum at t={ik_times[rh_min_idx]:.2f}s")
print(f"  Offset: {abs(ik_times[lh_min_idx] - ik_times[rh_min_idx]):.2f}s")

if abs(ik_times[lh_min_idx] - ik_times[rh_min_idx]) > 1.0:
    print(f"  ✓ Legs are ALTERNATING (good for walking)")
else:
    print(f"  ✗ Legs are SYNCHRONIZED (not alternating - no walking!)")

print("\n" + "="*70)
