#!/usr/bin/env python3
"""
Test: Run IK for 5 consecutive frames without viewer
Check if base Z stays at 0.2m or resets to 0.1m
"""

import mujoco
import numpy as np

def load_model():
    model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
    return model

def compute_ik_solution(model, data, base_target_pos, com_target_pos, foot1_target_pos, foot2_target_pos, max_iterations=50, tolerance=0.002):
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
            qpos[6 + i] = np.clip(qpos[6 + i] + dq_all[3 + i], model.jnt_range[i, 0], model.jnt_range[i, 1])
        
        qpos[3:7] = [1, 0, 0, 0]
    
    return qpos, False

model = load_model()
data = mujoco.MjData(model)

base_traj = np.load('base_trajectory.npy')
com_traj = np.load('com_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

com_id = model.site(name='com_site').id
f1_id = model.site(name='foot1_site').id
f2_id = model.site(name='foot2_site').id

print("\n=== SEQUENTIAL IK TEST (5 frames, step 100-104) ===\n")
print("Frame | data.qpos[2] | qpos_sol[2] | smooth[2] | Foot1 Z | Foot2 Z")
print("------|------|----------|-----------|---------|---------|")

prev_qpos = data.qpos.copy()

for frame in range(100, 105):
    qpos_sol, success = compute_ik_solution(
        model, data,
        base_traj[frame],
        com_traj[frame],
        foot1_traj[frame],
        foot2_traj[frame],
        max_iterations=50,
        tolerance=0.002
    )
    
    smoothed = 0.7 * qpos_sol + 0.3 * prev_qpos
    data.qpos[:] = smoothed
    mujoco.mj_forward(model, data)
    
    print(f"{frame:5d} | {data.qpos[2]:.6f} | {qpos_sol[2]:.6f} | {smoothed[2]:.6f} | {data.site_xpos[f1_id,2]:.6f} | {data.site_xpos[f2_id,2]:.6f}")
    
    prev_qpos = smoothed.copy()

print("\nExpected: All values should be around 0.2m for base Z and 0.21-0.225m for feet")
