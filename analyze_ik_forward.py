#!/usr/bin/env python3
"""
Create forward walking trajectory by adding forward offset to IK leg trajectories
"""

import numpy as np

# Load IK trajectories
ik_times = np.load("ik_times.npy")
ik_left_hip = np.load("ik_left_hip.npy")
ik_left_knee = np.load("ik_left_knee.npy")
ik_left_ankle = np.load("ik_left_ankle.npy")
ik_right_hip = np.load("ik_right_hip.npy")
ik_right_knee = np.load("ik_right_knee.npy")
ik_right_ankle = np.load("ik_right_ankle.npy")

print("Original IK trajectory info:")
print(f"Times: {ik_times[0]:.2f} to {ik_times[-1]:.2f}s (period: {ik_times[-1]-ik_times[0]:.2f}s)")
print(f"Left hip (degrees): {np.degrees(ik_left_hip).min():.1f} to {np.degrees(ik_left_hip).max():.1f}")
print(f"Left knee (degrees): {np.degrees(ik_left_knee).min():.1f} to {np.degrees(ik_left_knee).max():.1f}")
print(f"Left ankle (degrees): {np.degrees(ik_left_ankle).min():.1f} to {np.degrees(ik_left_ankle).max():.1f}")
print()

# The issue: IK trajectories are leg-only (hip/knee/ankle), no forward locomotion
# Solution: These are relative joint angles. To make forward walking:
# 1. Keep the IK leg motions at reduced amplitude
# 2. Add forward velocity to BASE (root_x), not to individual legs
# 3. The leg motions provide stepping, base provides propulsion

# Actually the real problem: we need to check if there's FORWARD MOTION in the IK trajectory
# Look at hip position change over cycle

# For a forward walking gait, each foot should swing forward during swing phase
# The current IK is probably symmetric (left and right do same thing at different times)

# Let's examine the pattern
gait_period = ik_times[-1]
print(f"Analyzing gait for {len(ik_times)} points over {gait_period:.2f}s")

# Check which leg is which phase
print("\nAt key cycle points:")
for i, t_frac in enumerate([0.0, 0.25, 0.5, 0.75]):
    idx = int(len(ik_times) * t_frac)
    print(f"Time {t_frac*100:.0f}% (idx {idx}): L_hip={np.degrees(ik_left_hip[idx]):6.1f}° L_knee={np.degrees(ik_left_knee[idx]):6.1f}° L_ankle={np.degrees(ik_left_ankle[idx]):6.1f}° | R_hip={np.degrees(ik_right_hip[idx]):6.1f}° R_knee={np.degrees(ik_right_knee[idx]):6.1f}° R_ankle={np.degrees(ik_right_ankle[idx]):6.1f}°")

print("\n" + "="*80)
print("KEY FINDING: IK trajectories contain STEPPING (leg flexion/extension)")
print("but NO FORWARD LOCOMOTION (no forward foot placement)")
print("="*80)
print("\nTo achieve forward walking, we need:")
print("1. Apply IK trajectories for STEPPING motion (25% amplitude)")
print("2. Add external FORWARD FORCE on base (not tracked position)")
print("3. This matches the previous successful test with walk_fixed_height.py")
print("\nThe robot legs are STEPPING but body isn't MOVING FORWARD")
print("because gait creates balanced forces (push back as much as forward)")
print("\nSolution: Use forward force that's NOT fighting the gait dynamics")
