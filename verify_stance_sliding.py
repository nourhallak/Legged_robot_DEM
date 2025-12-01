#!/usr/bin/env python3
"""
Verify feet don't slide forward/backward during stance phases.
"""
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
foot1_traj = np.load(os.path.join(script_dir, "foot1_trajectory.npy"))
foot2_traj = np.load(os.path.join(script_dir, "foot2_trajectory.npy"))

print("=== STANCE PHASE SLIDING ANALYSIS ===\n")

# Left foot stance: frames 0-100, 200-300
print("LEFT FOOT DURING STANCE (should stay at same X):")
print("\nFrames 0-100 (first half, should be at X ≈ 0.0):")
left_stance_1 = foot1_traj[0:100, 0]
print(f"  X range: {left_stance_1.min():.6f} to {left_stance_1.max():.6f} m")
print(f"  X std dev: {left_stance_1.std():.9f} m")
print(f"  Max variation: {left_stance_1.max() - left_stance_1.min():.9f} m")

print("\nFrames 200-300 (second cycle first half, should be at X ≈ 0.0):")
left_stance_2 = foot1_traj[200:300, 0]
print(f"  X range: {left_stance_2.min():.6f} to {left_stance_2.max():.6f} m")
print(f"  X std dev: {left_stance_2.std():.9f} m")
print(f"  Max variation: {left_stance_2.max() - left_stance_2.min():.9f} m")

# Right foot stance: frames 100-200, 300-400
print("\nRIGHT FOOT DURING STANCE (should stay at same X):")
print("\nFrames 100-200 (first cycle second half, should be at X ≈ 0.06):")
right_stance_1 = foot2_traj[100:200, 0]
print(f"  X range: {right_stance_1.min():.6f} to {right_stance_1.max():.6f} m")
print(f"  X std dev: {right_stance_1.std():.9f} m")
print(f"  Max variation: {right_stance_1.max() - right_stance_1.min():.9f} m")

print("\nFrames 300-400 (second cycle second half, should be at X ≈ 0.06):")
right_stance_2 = foot2_traj[300:400, 0]
print(f"  X range: {right_stance_2.min():.6f} to {right_stance_2.max():.6f} m")
print(f"  X std dev: {right_stance_2.std():.9f} m")
print(f"  Max variation: {right_stance_2.max() - right_stance_2.min():.9f} m")

print("\n=== SWING PHASE FORWARD PROGRESSION ===\n")

# Left foot swing: frames 100-200 (swing from 0 to 0.12)
print("LEFT FOOT SWING (should progress from X ≈ 0.0 to X ≈ 0.12):")
left_swing = foot1_traj[100:200, 0]
print(f"  X range: {left_swing.min():.6f} to {left_swing.max():.6f} m")
print(f"  Forward progress: {left_swing.max() - left_swing.min():.6f} m (should be ~0.12)")

print("\nRIGHT FOOT SWING (should progress from X ≈ 0.06 to X ≈ 0.18):")
right_swing = foot2_traj[0:100, 0]
print(f"  X range: {right_swing.min():.6f} to {right_swing.max():.6f} m")
print(f"  Forward progress: {right_swing.max() - right_swing.min():.6f} m (should be ~0.12)")

print("\n=== VERIFICATION ===")

stance_ok = (left_stance_1.std() < 1e-6 and left_stance_2.std() < 1e-6 and 
             right_stance_1.std() < 1e-6 and right_stance_2.std() < 1e-6)

if stance_ok:
    print("✓ SUCCESS: Stance feet do NOT slide (stay fixed in X)")
else:
    print("✗ ERROR: Stance feet are sliding in X!")

swing_ok = (left_swing.max() - left_swing.min() > 0.10 and 
            right_swing.max() - right_swing.min() > 0.10)

if swing_ok:
    print("✓ SUCCESS: Swing feet progress forward properly")
else:
    print("✗ ERROR: Swing feet not progressing properly!")
