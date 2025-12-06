#!/usr/bin/env python3
"""
Debug IK convergence by testing a single point in detail
"""

import numpy as np
import mujoco
import re
from pathlib import Path

def load_model_with_assets(xml_path="legged_robot_ik.xml"):
    """Load MuJoCo model with correct asset paths"""
    
    xml_file = Path(xml_path)
    if not xml_file.exists():
        xml_file = Path(".") / "legged_robot_ik.xml"
    
    with open(xml_file, "r") as f:
        xml_content = f.read()
    
    # Update mesh paths
    asset_dir = xml_file.parent / "Legged_robot" / "meshes"
    
    for match in re.finditer(r'filename="([^"]*\.stl)"', xml_content):
        stl_file = match.group(1)
        full_path = asset_dir / stl_file
        
        if full_path.exists():
            xml_content = xml_content.replace(f'filename="{stl_file}"', f'filename="{full_path}"')
    
    return mujoco.MjModel.from_xml_string(xml_content)

print("="*80)
print("IK CONVERGENCE DEBUG - SINGLE POINT TEST")
print("="*80)

# Load model
model = load_model_with_assets()
data = mujoco.MjData(model)

# Get site/body IDs
foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id
base_body_id = model.body(name='hip').id

# Load trajectories
base_traj = np.load("base_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

# Test on first trajectory point
step = 0
base_target = base_traj[step]
foot1_target = foot1_traj[step]
foot2_target = foot2_traj[step]

print(f"\nTarget positions (step {step}):")
print(f"  Base target:  {base_target}")
print(f"  Foot1 target: {foot1_target}")
print(f"  Foot2 target: {foot2_target}")

# Start from zero pose
qpos = np.zeros(model.nq)
qpos[2] = 0.42  # hip height

print(f"\nInitial qpos: {qpos}")

# Forward kinematics
data.qpos[:] = qpos
mujoco.mj_forward(model, data)

base_pos = data.xpos[base_body_id]
foot1_pos = data.site_xpos[foot1_site_id]
foot2_pos = data.site_xpos[foot2_site_id]

print(f"\nInitial positions (FK):")
print(f"  Base actual:  {base_pos}")
print(f"  Foot1 actual: {foot1_pos}")
print(f"  Foot2 actual: {foot2_pos}")

# Check errors
base_error = base_target - base_pos
foot1_error = foot1_target - foot1_pos
foot2_error = foot2_target - foot2_pos

print(f"\nInitial errors:")
print(f"  Base error:  {base_error}, norm={np.linalg.norm(base_error):.6f}")
print(f"  Foot1 error: {foot1_error}, norm={np.linalg.norm(foot1_error):.6f}")
print(f"  Foot2 error: {foot2_error}, norm={np.linalg.norm(foot2_error):.6f}")
print(f"  Total error: {np.linalg.norm(base_error) + np.linalg.norm(foot1_error) + np.linalg.norm(foot2_error):.6f}")

# Compute Jacobian
print(f"\nComputing Jacobian...")
act_joints = [3, 4, 5, 6, 7, 8]
n_act_joints = len(act_joints)
epsilon = 1e-3

jacobian = np.zeros((9, n_act_joints))

for j_idx, j in enumerate(act_joints):
    # Perturb joint j
    qpos_plus = qpos.copy()
    qpos_plus[j] += epsilon
    
    data.qpos[:] = qpos_plus
    mujoco.mj_forward(model, data)
    
    base_pos_plus = data.xpos[base_body_id]
    foot1_pos_plus = data.site_xpos[foot1_site_id]
    foot2_pos_plus = data.site_xpos[foot2_site_id]
    
    jacobian[0:3, j_idx] = (base_pos_plus - base_pos) / epsilon
    jacobian[3:6, j_idx] = (foot1_pos_plus - foot1_pos) / epsilon
    jacobian[6:9, j_idx] = (foot2_pos_plus - foot2_pos) / epsilon

print(f"Jacobian shape: {jacobian.shape}")
print(f"Jacobian rank: {np.linalg.matrix_rank(jacobian)}")
print(f"Jacobian condition number: {np.linalg.cond(jacobian):.2e}")

print(f"\nJacobian:")
print(jacobian)

# Compute DLS pseudo-inverse
damping = 1.0
J_T = jacobian.T
H = J_T @ jacobian + damping * np.eye(n_act_joints)
print(f"\nH matrix condition number: {np.linalg.cond(H):.2e}")

try:
    H_inv = np.linalg.inv(H)
    jacobian_dls = H_inv @ J_T
    print(f"DLS inverse computed successfully")
    print(f"DLS Jacobian shape: {jacobian_dls.shape}")
except Exception as e:
    print(f"ERROR computing DLS inverse: {e}")
    jacobian_dls = None

# Test IK update
if jacobian_dls is not None:
    errors = np.concatenate([base_error, foot1_error, foot2_error])
    learning_rate = 0.05
    dq_act = learning_rate * jacobian_dls @ errors
    
    print(f"\nIK update:")
    print(f"  Errors: {errors}")
    print(f"  dq_act: {dq_act}")
    print(f"  dq_act magnitude: {np.linalg.norm(dq_act):.6f}")
    
    # Apply update
    qpos_new = qpos.copy()
    for j_idx, j in enumerate(act_joints):
        qpos_new[j] += dq_act[j_idx]
    
    # Apply limits
    for j in act_joints:
        if j < model.jnt_range.shape[0]:
            qmin, qmax = model.jnt_range[j]
            qpos_new[j] = np.clip(qpos_new[j], qmin, qmax)
    
    print(f"\nNew qpos: {qpos_new}")
    
    # Check new positions
    data.qpos[:] = qpos_new
    mujoco.mj_forward(model, data)
    
    base_pos_new = data.xpos[base_body_id]
    foot1_pos_new = data.site_xpos[foot1_site_id]
    foot2_pos_new = data.site_xpos[foot2_site_id]
    
    print(f"\nNew positions (after 1 IK iteration):")
    print(f"  Base actual:  {base_pos_new}")
    print(f"  Foot1 actual: {foot1_pos_new}")
    print(f"  Foot2 actual: {foot2_pos_new}")
    
    base_error_new = base_target - base_pos_new
    foot1_error_new = foot1_target - foot1_pos_new
    foot2_error_new = foot2_target - foot2_pos_new
    
    print(f"\nNew errors:")
    print(f"  Base error:  norm={np.linalg.norm(base_error_new):.6f}")
    print(f"  Foot1 error: norm={np.linalg.norm(foot1_error_new):.6f}")
    print(f"  Foot2 error: norm={np.linalg.norm(foot2_error_new):.6f}")
    print(f"  Total error: {np.linalg.norm(base_error_new) + np.linalg.norm(foot1_error_new) + np.linalg.norm(foot2_error_new):.6f}")

print("\n" + "="*80)
