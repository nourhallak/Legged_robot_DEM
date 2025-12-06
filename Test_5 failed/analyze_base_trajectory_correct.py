#!/usr/bin/env python3
"""
Correct Base (Hip) Trajectory Error Analysis

Properly handles qpos offset vs world coordinates.
"""

import numpy as np
import mujoco

print("\n" + "="*80)
print("BASE (HIP) TRAJECTORY ERROR - DETAILED ANALYSIS")
print("="*80)

# Load model and data
model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
data = mujoco.MjData(model)

# Load trajectories
print("\nLoading data...")
base_qpos = np.load("base_feasible.npy")  # These are qpos values (offsets)
q_left = np.load("q_left_feasible.npy")
q_right = np.load("q_right_feasible.npy")

NUM_STEPS = len(q_left)
print(f"  Loaded {NUM_STEPS} trajectory points")

# Constants from XML
HIP_INIT_Z = 0.42  # From XML: <body name="hip" pos="0 -0.0049659 0.42">
HIP_INIT_Y = -0.0049659

print(f"\nHip body initial position (from XML):")
print(f"  X: 0.0 mm")
print(f"  Y: {HIP_INIT_Y*1000:.4f} mm")
print(f"  Z: {HIP_INIT_Z*1000:.1f} mm")

# Get actual hip positions from forward kinematics
print("\nCalculating actual hip world positions from joint angles...")
hip_actual_world = np.zeros((NUM_STEPS, 3))
hip_actual_qpos = np.zeros((NUM_STEPS, 3))

for step in range(NUM_STEPS):
    # Set qpos
    data.qpos[0:3] = base_qpos[step]  # Base qpos (offsets)
    data.qpos[3:6] = q_left[step]
    data.qpos[6:9] = q_right[step]
    
    mujoco.mj_forward(model, data)
    
    # Get actual hip position in world coordinates
    hip_actual_world[step] = data.xpos[1].copy()
    hip_actual_qpos[step] = data.qpos[0:3].copy()

print("✓ Actual positions calculated")

# Calculate what the world position SHOULD be
print("\nCalculating expected world positions...")
# Expected world position = body initial position + qpos offset
hip_expected_world = np.zeros((NUM_STEPS, 3))
hip_expected_world[:, 0] = base_qpos[:, 0]  # X offset becomes world X
hip_expected_world[:, 1] = HIP_INIT_Y + base_qpos[:, 1]  # Initial Y + offset
hip_expected_world[:, 2] = HIP_INIT_Z + base_qpos[:, 2]  # Initial Z + offset

print(f"\nExpected world Z range: {(HIP_INIT_Z + base_qpos[:, 2].min())*1000:.1f} to {(HIP_INIT_Z + base_qpos[:, 2].max())*1000:.1f} mm")
print(f"Actual world Z range:   {hip_actual_world[:, 2].min()*1000:.1f} to {hip_actual_world[:, 2].max()*1000:.1f} mm")

# Calculate errors
err_hip = hip_actual_world - hip_expected_world
err_hip_mag = np.linalg.norm(err_hip, axis=1)

print(f"\nBase (Hip) World Position Error:")
print(f"  Mean magnitude: {err_hip_mag.mean()*1000:.3f} mm")
print(f"  Max magnitude:  {err_hip_mag.max()*1000:.3f} mm")
print(f"  Min magnitude:  {err_hip_mag.min()*1000:.3f} mm")

print(f"\nComponent Errors:")
print(f"  X: mean={err_hip[:, 0].mean()*1000:+.3f}mm, max={np.abs(err_hip[:, 0]).max()*1000:.3f}mm")
print(f"  Y: mean={err_hip[:, 1].mean()*1000:+.3f}mm, max={np.abs(err_hip[:, 1]).max()*1000:.3f}mm")
print(f"  Z: mean={err_hip[:, 2].mean()*1000:+.3f}mm, max={np.abs(err_hip[:, 2]).max()*1000:.3f}mm")

# Check if errors are acceptable
acceptable_threshold = 0.005  # 5mm
pct_acceptable = np.sum(err_hip_mag < acceptable_threshold) / NUM_STEPS * 100

print(f"\nAcceptability:")
print(f"  Points < 5mm: {np.sum(err_hip_mag < acceptable_threshold)}/{NUM_STEPS} ({pct_acceptable:.1f}%)")

# Detailed trajectory info
print(f"\nTrajectory Information:")
print(f"  Base qpos X range: {base_qpos[:, 0].min()*1000:.1f} to {base_qpos[:, 0].max()*1000:.1f} mm")
print(f"  Base qpos Y range: {base_qpos[:, 1].min()*1000:.4f} to {base_qpos[:, 1].max()*1000:.4f} mm")
print(f"  Base qpos Z range: {base_qpos[:, 2].min()*1000:.3f} to {base_qpos[:, 2].max()*1000:.3f} mm (offset)")
print(f"  Hip world Z range (expected): {hip_expected_world[:, 2].min()*1000:.1f} to {hip_expected_world[:, 2].max()*1000:.1f} mm")
print(f"  Hip world Z range (actual):   {hip_actual_world[:, 2].min()*1000:.1f} to {hip_actual_world[:, 2].max()*1000:.1f} mm")

print("\n" + "="*80)

if err_hip_mag.mean() < 0.001:
    print("✓✓✓ EXCELLENT - Base trajectory perfectly tracked ✓✓✓")
elif err_hip_mag.mean() < 0.005:
    print("✓✓ GOOD - Base trajectory well tracked ✓✓")
elif pct_acceptable > 90:
    print("✓ ACCEPTABLE - Most of trajectory within 5mm ✓")
else:
    print("⚠ WARNING - Base trajectory has significant errors ⚠")

print("="*80 + "\n")
