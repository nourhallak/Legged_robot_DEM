#!/usr/bin/env python3
"""
Run 5mm stride IK simulation for first 50 steps
Compare desired vs actual foot positions from ik_simulation.py's actual IK solver
"""

import mujoco
import mujoco.viewer
import numpy as np
import os
import re

# Load model
def load_model_with_assets():
    xml_path = 'legged_robot_ik.xml'
    model = mujoco.MjModel.from_xml_path(xml_path)
    return model

# IK solver (copied from ik_simulation.py)
def compute_ik_solution(model, data, base_target_pos, com_target_pos, foot1_target_pos, foot2_target_pos, max_iterations=20, tolerance=0.005):
    try:
        com_site_id = model.site(name='com_site').id
        foot1_site_id = model.site(name='foot1_site').id
        foot2_site_id = model.site(name='foot2_site').id
    except Exception as e:
        raise ValueError(f"Could not find required sites: {e}")
    
    qpos_init = data.qpos.copy()
    qpos = qpos_init.copy()
    
    alpha = 0.08
    epsilon = 1e-6
    
    for iteration in range(max_iterations):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        
        base_pos = qpos[0:3].copy()
        com_pos = data.site_xpos[com_site_id].copy()
        foot1_pos = data.site_xpos[foot1_site_id].copy()
        foot2_pos = data.site_xpos[foot2_site_id].copy()
        
        base_error = base_target_pos - base_pos
        com_error = com_target_pos - com_pos
        foot1_error = foot1_target_pos - foot1_pos
        foot2_error = foot2_target_pos - foot2_pos
        
        total_error = (np.linalg.norm(base_error) + np.linalg.norm(com_error) + 
                      np.linalg.norm(foot1_error) + np.linalg.norm(foot2_error))
        
        if total_error < tolerance:
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
        
        try:
            jacobian_pinv = np.linalg.pinv(jacobian, rcond=1e-6)
        except:
            return qpos, False
        
        errors = np.concatenate([base_error, com_error, foot1_error, foot2_error])
        
        dq_all = alpha * jacobian_pinv @ errors
        
        if np.any(~np.isfinite(dq_all)):
            return qpos, False
        
        dq_all = np.clip(dq_all, -0.2, 0.2)
        
        qpos[0:3] += dq_all[0:3]
        
        for i in range(6):
            qpos[6 + i] = qpos[6 + i] + dq_all[3 + i]
        
        for i in range(6):
            joint_idx = i
            if joint_idx < model.jnt_range.shape[0]:
                qpos[6 + i] = np.clip(qpos[6 + i], model.jnt_range[joint_idx, 0], model.jnt_range[joint_idx, 1])
        
        qpos[3:7] = [1, 0, 0, 0]
    
    return qpos, False

# Main test
model = load_model_with_assets()
data = mujoco.MjData(model)

base_traj = np.load('base_trajectory.npy')
com_traj = np.load('com_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

try:
    com_site_id = model.site(name='com_site').id
    foot1_site_id = model.site(name='foot1_site').id
    foot2_site_id = model.site(name='foot2_site').id
except:
    print("Error: Could not find sites")
    exit(1)

print("\n=== TESTING 5MM STRIDE WITH ACTUAL IK SOLVER ===\n")

qpos = data.qpos.copy()

for step in range(0, 50, 10):
    qpos, success = compute_ik_solution(
        model, data,
        base_traj[step],
        com_traj[step],
        foot1_traj[step],
        foot2_traj[step],
        max_iterations=50,
        tolerance=0.002
    )
    
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    foot1_pos = data.site_xpos[foot1_site_id]
    foot2_pos = data.site_xpos[foot2_site_id]
    
    foot1_x_error_mm = (foot1_traj[step, 0] - foot1_pos[0]) * 1000
    foot2_x_error_mm = (foot2_traj[step, 0] - foot2_pos[0]) * 1000
    foot1_z_error_mm = (foot1_traj[step, 2] - foot1_pos[2]) * 1000
    foot2_z_error_mm = (foot2_traj[step, 2] - foot2_pos[2]) * 1000
    
    print(f"Step {step:2d}: Foot1 X_err={foot1_x_error_mm:6.2f}mm Z_err={foot1_z_error_mm:6.2f}mm  |  " +
          f"Foot2 X_err={foot2_x_error_mm:6.2f}mm Z_err={foot2_z_error_mm:6.2f}mm")

print("\n✓ If X errors < 3mm, sliding is minimal")
