#!/usr/bin/env python3
"""
Debug why IK isn't reaching trajectory targets
"""

import mujoco
import numpy as np
from pathlib import Path

# Load model
model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
data = mujoco.MjData(model)

# Load trajectories
base_traj = np.load("base_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")
ik_solutions = np.load("joint_solutions_ik.npy")

print("="*80)
print("IK CONVERGENCE ANALYSIS")
print("="*80)

# Test a few key frames
test_frames = [0, 25, 50, 75, 100]

for frame_idx in test_frames:
    print(f"\nFrame {frame_idx}:")
    print("-" * 40)
    
    # Get planned positions
    base_pos = base_traj[frame_idx]
    foot1_target = foot1_traj[frame_idx]
    foot2_target = foot2_traj[frame_idx]
    
    # Set IK solution
    qpos = ik_solutions[frame_idx]
    
    # Evaluate forward kinematics with this solution
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    # Get actual foot positions (body positions)
    foot1_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_1")
    foot2_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_2")
    
    foot1_actual = data.body(foot1_body_id).xpos.copy()
    foot2_actual = data.body(foot2_body_id).xpos.copy()
    
    # Compute errors
    error1 = np.linalg.norm(foot1_actual - foot1_target)
    error2 = np.linalg.norm(foot2_actual - foot2_target)
    
    print(f"  Foot1:")
    print(f"    Target Z: {foot1_target[2]:.4f}m")
    print(f"    Actual Z: {foot1_actual[2]:.4f}m")
    print(f"    Z error: {foot1_actual[2] - foot1_target[2]:.4f}m")
    print(f"    Total error: {error1:.4f}m")
    
    print(f"  Foot2:")
    print(f"    Target Z: {foot2_target[2]:.4f}m")
    print(f"    Actual Z: {foot2_actual[2]:.4f}m")
    print(f"    Z error: {foot2_actual[2] - foot2_target[2]:.4f}m")
    print(f"    Total error: {error2:.4f}m")
    
    # Show Z heights
    print(f"\n  Z Heights:")
    print(f"    Foot1 target lift: {foot1_target[2] - 0.43:.4f}m ({(foot1_target[2] - 0.43)*1000:.1f}mm)")
    print(f"    Foot1 actual lift: {foot1_actual[2] - 0.43:.4f}m ({(foot1_actual[2] - 0.43)*1000:.1f}mm)")

print("\n" + "="*80)
print("WORKSPACE CHECK")
print("="*80)

# Check if feet can reach higher
print("\nTesting if feet can reach higher Z values...")
test_z_values = [0.43, 0.435, 0.44, 0.445, 0.45]

for target_z in test_z_values:
    # Try to reach point with foot1
    target_foot1 = foot1_traj[0].copy()
    target_foot1[2] = target_z
    
    # Simple IK: target current foot1 position but at higher Z
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    
    foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot1")
    current_foot1 = data.body(foot1_id).xpos.copy()
    
    print(f"  Target Z={target_z:.4f}m: ", end="")
    if target_z <= 0.44:
        print("REACHABLE (within typical range)")
    elif target_z <= 0.45:
        print("MARGINAL (close to limit)")
    else:
        print("UNREACHABLE (exceeds workspace)")
