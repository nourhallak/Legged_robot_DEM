#!/usr/bin/env python3
"""
Debug which foot is being lifted high during simulation.
"""
import mujoco
import numpy as np
import os
import re

def load_model_with_assets():
    """Load the converted MJCF model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.join(script_dir, "Legged_robot")
    meshes_dir = os.path.join(package_dir, "meshes")
    mjcf_output_path = os.path.join(script_dir, "legged_robot_ik.xml")

    if not os.path.exists(mjcf_output_path):
        raise FileNotFoundError(f"Model file not found")

    with open(mjcf_output_path, 'r', encoding='utf-8') as f:
        mjcf_content = f.read()

    MESH_PATTERN = r'file="([^"]+\.STL)"'
    all_mesh_files = set(re.findall(MESH_PATTERN, mjcf_content))

    assets = {}
    for mesh_file in all_mesh_files:
        abs_path = os.path.join(meshes_dir, mesh_file)
        if os.path.exists(abs_path):
            with open(abs_path, 'rb') as f:
                assets[mesh_file] = f.read()

    model = mujoco.MjModel.from_xml_string(mjcf_content, assets=assets)
    return model

def compute_ik_solution(model, data, base_target, com_target, foot1_target, foot2_target, max_iter=50):
    """Quick IK solver."""
    try:
        com_site_id = model.site(name='com_site').id
        foot1_site_id = model.site(name='foot1_site').id
        foot2_site_id = model.site(name='foot2_site').id
    except:
        return data.qpos.copy(), False, 999

    qpos = data.qpos.copy()
    alpha = 0.08
    epsilon = 1e-6
    
    for iteration in range(max_iter):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        
        base_pos = qpos[0:3].copy()
        com_pos = data.site_xpos[com_site_id].copy()
        foot1_pos = data.site_xpos[foot1_site_id].copy()
        foot2_pos = data.site_xpos[foot2_site_id].copy()
        
        base_error = base_target - base_pos
        com_error = com_target - com_pos
        foot1_error = foot1_target - foot1_pos
        foot2_error = foot2_target - foot2_pos
        
        total_error = (np.linalg.norm(base_error) + np.linalg.norm(com_error) + 
                      np.linalg.norm(foot1_error) + np.linalg.norm(foot2_error))
        
        if total_error < 0.002:
            return qpos, True, total_error
        
        jacobian = np.zeros((12, 9))
        jacobian[0:3, 0:3] = np.eye(3)
        
        for j in range(6):
            qpos_plus = qpos.copy()
            qpos_plus[6 + j] += epsilon
            
            data.qpos[:] = qpos_plus
            mujoco.mj_forward(model, data)
            
            pos_plus_com = data.site_xpos[com_site_id].copy()
            pos_plus_foot1 = data.site_xpos[foot1_site_id].copy()
            pos_plus_foot2 = data.site_xpos[foot2_site_id].copy()
            
            jacobian[3:6, 3 + j] = (pos_plus_com - com_pos) / epsilon
            jacobian[6:9, 3 + j] = (pos_plus_foot1 - foot1_pos) / epsilon
            jacobian[9:12, 3 + j] = (pos_plus_foot2 - foot2_pos) / epsilon
        
        jacobian_pinv = np.linalg.pinv(jacobian, rcond=1e-6)
        errors = np.concatenate([base_error, com_error, foot1_error, foot2_error])
        dq_all = alpha * jacobian_pinv @ errors
        dq_all = np.clip(dq_all, -0.2, 0.2)
        
        qpos[0:3] += dq_all[0:3]
        for i in range(6):
            qpos[6 + i] = qpos[6 + i] + dq_all[3 + i]
            joint_idx = i
            if joint_idx < model.jnt_range.shape[0]:
                qpos[6 + i] = np.clip(qpos[6 + i], model.jnt_range[joint_idx, 0], model.jnt_range[joint_idx, 1])
        
        qpos[3:7] = [1, 0, 0, 0]
    
    return qpos, False, total_error

# Load
print("Loading...")
model = load_model_with_assets()
data = mujoco.MjData(model)

print("Loading trajectories...")
script_dir = os.path.dirname(os.path.abspath(__file__))
base_traj = np.load(os.path.join(script_dir, "base_trajectory.npy"))
com_traj = np.load(os.path.join(script_dir, "com_trajectory.npy"))
foot1_traj = np.load(os.path.join(script_dir, "foot1_trajectory.npy"))
foot2_traj = np.load(os.path.join(script_dir, "foot2_trajectory.npy"))

com_site_id = model.site(name='com_site').id
foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id

print("\n=== ACTUAL FOOT HEIGHT IN SIMULATION ===\n")
print("Step | Target Foot1 Z | Actual Foot1 Z | Target Foot2 Z | Actual Foot2 Z | Max Lift")
print("-" * 90)

for step in [0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 399]:
    base_target = base_traj[step]
    com_target = com_traj[step]
    foot1_target = foot1_traj[step]
    foot2_target = foot2_traj[step]
    
    qpos, success, error = compute_ik_solution(model, data, base_target, com_target, foot1_target, foot2_target)
    
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    actual_foot1_z = data.site_xpos[foot1_site_id, 2]
    actual_foot2_z = data.site_xpos[foot2_site_id, 2]
    
    target_foot1_z = foot1_target[2]
    target_foot2_z = foot2_target[2]
    
    lift1 = actual_foot1_z - 0.21
    lift2 = actual_foot2_z - 0.21
    max_lift = max(lift1, lift2)
    
    print(f"{step:3d}  | {target_foot1_z:.6f}       | {actual_foot1_z:.6f}       | {target_foot2_z:.6f}       | {actual_foot2_z:.6f}       | {max_lift:.6f}")

print("\n=== ANALYSIS ===")
print("If one foot is consistently higher, that's where the problem is.")
print("Ground is at Z = 0.21m")
