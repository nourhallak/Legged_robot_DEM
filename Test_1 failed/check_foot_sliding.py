#!/usr/bin/env python3
"""
Check if feet are actually sliding or if they're just moving with the base.
"""
import mujoco
import numpy as np
import os
import re

def load_model_with_assets():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.join(script_dir, "Legged_robot")
    meshes_dir = os.path.join(package_dir, "meshes")
    mjcf_output_path = os.path.join(script_dir, "legged_robot_ik.xml")

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
    try:
        com_site_id = model.site(name='com_site').id
        foot1_site_id = model.site(name='foot1_site').id
        foot2_site_id = model.site(name='foot2_site').id
    except:
        return data.qpos.copy(), False

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
            return qpos, True
        
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
        
        # NO WEIGHTED ERRORS - use full foot errors
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
    
    return qpos, False

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

print("\n=== STANCE PHASE SLIDING CHECK ===\n")
print("Step | Target Foot1 X | Actual Foot1 X | Slip (mm) | Target Foot2 X | Actual Foot2 X | Slip (mm) | Phase")
print("-" * 105)

for step in [0, 20, 50, 80, 100, 120, 150, 180, 200]:
    base_target = base_traj[step]
    com_target = com_traj[step]
    foot1_target = foot1_traj[step]
    foot2_target = foot2_traj[step]
    
    qpos, _ = compute_ik_solution(model, data, base_target, com_target, foot1_target, foot2_target)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    actual_foot1_x = data.site_xpos[foot1_site_id, 0]
    actual_foot2_x = data.site_xpos[foot2_site_id, 0]
    
    target_foot1_x = foot1_target[0]
    target_foot2_x = foot2_target[0]
    
    slip1 = (actual_foot1_x - target_foot1_x) * 1000  # Convert to mm
    slip2 = (actual_foot2_x - target_foot2_x) * 1000
    
    # Determine phase
    cycle_pos = step % 200
    if cycle_pos < 100:
        phase_l = "STANCE"
        phase_r = "SWING"
    else:
        phase_l = "SWING"
        phase_r = "STANCE"
    
    phase = f"{phase_l} {phase_r}"
    
    print(f"{step:3d}  | {target_foot1_x:14.6f} | {actual_foot1_x:14.6f} | {slip1:9.2f} | {target_foot2_x:14.6f} | {actual_foot2_x:14.6f} | {slip2:9.2f} | {phase}")

print("\n=== ANALYSIS ===")
print("Slip > 0.5mm indicates sliding during stance phases.")
