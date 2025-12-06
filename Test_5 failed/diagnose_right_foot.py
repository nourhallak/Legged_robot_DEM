#!/usr/bin/env python3
"""
Diagnose Right Foot High Error

Check if right foot trajectory is within reachable workspace.
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt

print("\n" + "="*80)
print("DIAGNOSING RIGHT FOOT ERROR")
print("="*80)

model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
data = mujoco.MjData(model)

# Load trajectories
foot2_planned = np.load("foot2_feasible.npy")
q_right = np.load("q_right_feasible.npy")

print("\nAnalyzing right foot trajectory...")
print(f"Planned positions:")
print(f"  X: {foot2_planned[:, 0].min()*1000:.1f} to {foot2_planned[:, 0].max()*1000:.1f} mm")
print(f"  Y: {foot2_planned[:, 1].min()*1000:.1f} to {foot2_planned[:, 1].max()*1000:.1f} mm")
print(f"  Z: {foot2_planned[:, 2].min()*1000:.1f} to {foot2_planned[:, 2].max()*1000:.1f} mm")

# Get actual positions
foot2_actual = np.zeros_like(foot2_planned)
for step in range(len(q_right)):
    data.qpos[6:9] = q_right[step]
    mujoco.mj_forward(model, data)
    foot2_actual[step] = data.site_xpos[1].copy()

print(f"\nActual positions from IK:")
print(f"  X: {foot2_actual[:, 0].min()*1000:.1f} to {foot2_actual[:, 0].max()*1000:.1f} mm")
print(f"  Y: {foot2_actual[:, 1].min()*1000:.1f} to {foot2_actual[:, 1].max()*1000:.1f} mm")
print(f"  Z: {foot2_actual[:, 2].min()*1000:.1f} to {foot2_actual[:, 2].max()*1000:.1f} mm")

# Check workspace for right foot with different joint angles
print("\nScanning right foot workspace...")
positions_r = []
for hip in np.linspace(-np.pi/3, 0, 15):
    for knee in np.linspace(-np.pi/2, 0, 15):
        for ankle in [0.0]:  # Fixed ankle for flat foot
            data.qpos[6:9] = [hip, knee, ankle]
            mujoco.mj_forward(model, data)
            positions_r.append(data.site_xpos[1].copy())

positions_r = np.array(positions_r)
print(f"\nRight foot reachable workspace (flat foot, ankle=0):")
print(f"  X: {positions_r[:, 0].min()*1000:.1f} to {positions_r[:, 0].max()*1000:.1f} mm")
print(f"  Y: {positions_r[:, 1].min()*1000:.1f} to {positions_r[:, 1].max()*1000:.1f} mm")
print(f"  Z: {positions_r[:, 2].min()*1000:.1f} to {positions_r[:, 2].max()*1000:.1f} mm")

# Check if trajectory is in workspace
in_workspace_x = (foot2_planned[:, 0] >= positions_r[:, 0].min()) & (foot2_planned[:, 0] <= positions_r[:, 0].max())
in_workspace_y = (foot2_planned[:, 1] >= positions_r[:, 1].min()) & (foot2_planned[:, 1] <= positions_r[:, 1].max())
in_workspace_z = (foot2_planned[:, 2] >= positions_r[:, 2].min()) & (foot2_planned[:, 2] <= positions_r[:, 2].max())
in_workspace = in_workspace_x & in_workspace_y & in_workspace_z

print(f"\nTrajectory points in workspace:")
print(f"  X: {np.sum(in_workspace_x)}/{len(foot2_planned)}")
print(f"  Y: {np.sum(in_workspace_y)}/{len(foot2_planned)}")
print(f"  Z: {np.sum(in_workspace_z)}/{len(foot2_planned)}")
print(f"  All: {np.sum(in_workspace)}/{len(foot2_planned)}")

# Find problematic points
problem_indices = np.where(~in_workspace)[0]
if len(problem_indices) > 0:
    print(f"\n⚠ PROBLEM DETECTED: {len(problem_indices)} trajectory points out of workspace!")
    print(f"  First problem at step {problem_indices[0]}")
    print(f"  Planned: {foot2_planned[problem_indices[0]]*1000}")
    print(f"  Workspace Y: {positions_r[:, 1].min()*1000:.1f} to {positions_r[:, 1].max()*1000:.1f}")
else:
    print(f"\n✓ All trajectory points are within workspace")

# Analyze Y lateral position issue
print(f"\nLateral (Y) Position Analysis:")
print(f"  Right leg fixed Y: {positions_r[0, 1]*1000:.1f} mm")
print(f"  Planned Y: {foot2_planned[0, 1]*1000:.1f} mm")
print(f"  Difference: {(foot2_planned[0, 1] - positions_r[0, 1])*1000:.1f} mm")

err = np.linalg.norm(foot2_planned - foot2_actual, axis=1)
print(f"\nError statistics:")
print(f"  Mean: {err.mean()*1000:.1f} mm")
print(f"  Y-component mean error: {(foot2_planned[:, 1] - foot2_actual[:, 1]).mean()*1000:.1f} mm")

print("\n" + "="*80)
