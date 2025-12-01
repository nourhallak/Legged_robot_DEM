#!/usr/bin/env python3
"""Test IK WITHOUT smoothing to see if that's the issue"""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

base_traj = np.load('base_trajectory.npy')
com_traj = np.load('com_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

com_id = model.site(name='com_site').id
f1_id = model.site(name='foot1_site').id
f2_id = model.site(name='foot2_site').id

def compute_ik_solution(model, data, base_target, com_target, foot1_target, foot2_target, max_iterations=50, tolerance=0.002):
    """Exact copy from ik_simulation.py"""
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
        
        base_error = base_target - base_pos
        com_error = com_target - com_pos
        foot1_error = foot1_target - foot1_pos
        foot2_error = foot2_target - foot2_pos
        
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
            qpos[6+i] = np.clip(qpos[6+i] + dq_all[3+i], model.jnt_range[i,0], model.jnt_range[i,1])
        
        qpos[3:7] = [1, 0, 0, 0]
    
    return qpos, False

print("=== Testing IK WITHOUT smoothing ===\n")
print("Step | Base Z | Foot1 Z | Foot2 Z | Errors (mm)")
print("-" * 60)

qpos_current = data.qpos.copy()
flying_foot_count = 0

for step_index in range(len(base_traj)):
    base_target = base_traj[step_index]
    com_target = com_traj[step_index]
    foot1_target = foot1_traj[step_index]
    foot2_target = foot2_traj[step_index]
    
    # Set data to current position for IK to start from
    data.qpos[:] = qpos_current
    mujoco.mj_forward(model, data)
    
    qpos_current, success = compute_ik_solution(
        model, data,
        base_target, com_target, foot1_target, foot2_target,
        max_iterations=50, tolerance=0.002
    )
    
    # NO smoothing - just use IK result directly
    data.qpos[:] = qpos_current
    mujoco.mj_forward(model, data)
    
    # Get foot positions and errors
    f1_z = data.site_xpos[f1_id, 2] * 1000
    f2_z = data.site_xpos[f2_id, 2] * 1000
    target_f1 = foot1_target[2] * 1000
    target_f2 = foot2_target[2] * 1000
    f1_err = f1_z - target_f1
    f2_err = f2_z - target_f2
    
    # Print every 50 steps
    if step_index % 50 == 0 or step_index == len(base_traj) - 1:
        print(f"{step_index:3d} | {qpos_current[2]:6.4f} | {f1_z:7.1f} | {f2_z:7.1f} | F1:{f1_err:6.1f} F2:{f2_err:6.1f}")
    
    # Count flying feet
    if abs(f1_err) > 10:
        flying_foot_count += 1
    if abs(f2_err) > 10:
        flying_foot_count += 1

print(f"\nTotal foot errors > 10mm: {flying_foot_count}")
