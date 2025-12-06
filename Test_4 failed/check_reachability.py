#!/usr/bin/env python3
"""
Check robot reachability - can feet reach the trajectory points?
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
data = mujoco.MjData(model)

# Load trajectories
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

print("="*80)
print("TRAJECTORY FEASIBILITY CHECK")
print("="*80)

# Sample foot positions throughout trajectory
print("\nFoot1 trajectory stats:")
print(f"  X range: [{foot1_traj[:,0].min():.4f}, {foot1_traj[:,0].max():.4f}]m")
print(f"  Y range: [{foot1_traj[:,1].min():.4f}, {foot1_traj[:,1].max():.4f}]m")
print(f"  Z range: [{foot1_traj[:,2].min():.4f}, {foot1_traj[:,2].max():.4f}]m")

print("\nFoot2 trajectory stats:")
print(f"  X range: [{foot2_traj[:,0].min():.4f}, {foot2_traj[:,0].max():.4f}]m")
print(f"  Y range: [{foot2_traj[:,1].min():.4f}, {foot2_traj[:,1].max():.4f}]m")
print(f"  Z range: [{foot2_traj[:,2].min():.4f}, {foot2_traj[:,2].max():.4f}]m")

# Test reachability
print("\n" + "="*80)
print("CHECKING REACHABILITY")
print("="*80)

# Try to reach a few target positions with simple IK
from scipy.optimize import minimize

def reach_error(qpos, model, data, target_pos, site_id):
    """Compute distance from site to target"""
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    actual_pos = data.site_xpos[site_id].copy()
    return np.linalg.norm(actual_pos - target_pos)

foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id

# Test a few key positions
test_indices = [0, 100, 200, 300]

print("\nTesting reach for sample trajectory points:")
for idx in test_indices:
    print(f"\nFrame {idx}:")
    
    target1 = foot1_traj[idx]
    target2 = foot2_traj[idx]
    
    print(f"  Foot1 target: {target1}")
    print(f"  Foot2 target: {target2}")
    
    # Check current solution
    ik_solutions = np.load("joint_solutions_ik.npy")
    data.qpos[:] = ik_solutions[idx]
    mujoco.mj_forward(model, data)
    
    actual1 = data.site_xpos[foot1_site_id]
    actual2 = data.site_xpos[foot2_site_id]
    
    err1 = np.linalg.norm(actual1 - target1)
    err2 = np.linalg.norm(actual2 - target2)
    
    print(f"  Foot1 error: {err1*1000:.2f}mm")
    print(f"  Foot2 error: {err2*1000:.2f}mm")

# Check joint limits
print("\n" + "="*80)
print("JOINT LIMITS")
print("="*80)

for j in range(model.njnt):
    if j < model.jnt_range.shape[0]:
        qmin, qmax = model.jnt_range[j]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        print(f"  Joint {j} ({joint_name}): [{np.degrees(qmin):.1f}, {np.degrees(qmax):.1f}] deg")
