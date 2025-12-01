#!/usr/bin/env python3
"""Compare learning rates for IK with weighted errors"""
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

base_target = base_traj[0]
com_target = com_traj[0]
f1_target = foot1_traj[0]
f2_target = foot2_traj[0]

for alpha in [0.01, 0.02, 0.04, 0.08]:
    q = data.qpos.copy()
    
    for iteration in range(50):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        
        base_pos = q[0:3].copy()
        com_pos = data.site_xpos[com_id].copy()
        f1_pos = data.site_xpos[f1_id].copy()
        f2_pos = data.site_xpos[f2_id].copy()
        
        base_error = base_target - base_pos
        com_error = com_target - com_pos
        f1_error = f1_target - f1_pos
        f2_error = f2_target - f2_pos
        
        total_error = (np.linalg.norm(base_error) + np.linalg.norm(com_error) + 
                      np.linalg.norm(f1_error) + np.linalg.norm(f2_error))
        
        if total_error < 0.002:
            break
        
        jacobian = np.zeros((12, 9))
        jacobian[0:3, 0:3] = np.eye(3)
        
        for j in range(6):
            qpos_plus = q.copy()
            qpos_plus[6 + j] += 1e-6
            
            data.qpos[:] = qpos_plus
            mujoco.mj_forward(model, data)
            
            pos_plus_com = data.site_xpos[com_id].copy()
            pos_plus_foot1 = data.site_xpos[f1_id].copy()
            pos_plus_foot2 = data.site_xpos[f2_id].copy()
            
            jacobian[3:6, 3 + j] = (pos_plus_com - com_pos) / 1e-6
            jacobian[6:9, 3 + j] = (pos_plus_foot1 - f1_pos) / 1e-6
            jacobian[9:12, 3 + j] = (pos_plus_foot2 - f2_pos) / 1e-6
        
        # Normalize each target's error independently
        base_error_norm = base_error / (np.linalg.norm(base_error) + 1e-10)
        com_error_norm = com_error / (np.linalg.norm(com_error) + 1e-10)
        f1_error_norm = f1_error / (np.linalg.norm(f1_error) + 1e-10)
        f2_error_norm = f2_error / (np.linalg.norm(f2_error) + 1e-10)
        
        errors = np.concatenate([base_error_norm, com_error_norm, f1_error_norm, f2_error_norm])
        
        jacobian_pinv = np.linalg.pinv(jacobian, rcond=1e-6)
        dq_all = alpha * jacobian_pinv @ errors
        dq_all = np.clip(dq_all, -0.2, 0.2)
        
        q[0:3] += dq_all[0:3]
        for i in range(6):
            q[6 + i] = np.clip(q[6 + i] + dq_all[3 + i], 
                               model.jnt_range[i, 0], model.jnt_range[i, 1])
    
    print(f"Alpha={alpha:.2f}: Base Z={q[2]:.4f} (error={total_error:.6f}) after {iteration+1} iters")

print("\n\nNow try: Apply BOTH per-target weighting AND scale errors by target distance")

# Try weighting by target error magnitude
for alpha in [0.08]:
    q = data.qpos.copy()
    
    for iteration in range(50):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        
        base_pos = q[0:3].copy()
        com_pos = data.site_xpos[com_id].copy()
        f1_pos = data.site_xpos[f1_id].copy()
        f2_pos = data.site_xpos[f2_id].copy()
        
        base_error = base_target - base_pos
        com_error = com_target - com_pos
        f1_error = f1_target - f1_pos
        f2_error = f2_target - f2_pos
        
        total_error = (np.linalg.norm(base_error) + np.linalg.norm(com_error) + 
                      np.linalg.norm(f1_error) + np.linalg.norm(f2_error))
        
        if total_error < 0.002:
            break
        
        jacobian = np.zeros((12, 9))
        jacobian[0:3, 0:3] = np.eye(3)
        
        for j in range(6):
            qpos_plus = q.copy()
            qpos_plus[6 + j] += 1e-6
            
            data.qpos[:] = qpos_plus
            mujoco.mj_forward(model, data)
            
            pos_plus_com = data.site_xpos[com_id].copy()
            pos_plus_foot1 = data.site_xpos[f1_id].copy()
            pos_plus_foot2 = data.site_xpos[f2_id].copy()
            
            jacobian[3:6, 3 + j] = (pos_plus_com - com_pos) / 1e-6
            jacobian[6:9, 3 + j] = (pos_plus_foot1 - f1_pos) / 1e-6
            jacobian[9:12, 3 + j] = (pos_plus_foot2 - f2_pos) / 1e-6
        
        # Weight each error by 1/magnitude (scale errors inversely to their magnitude)
        # This makes small errors (foot positions close to target) count more
        w_base = 0.25 / (np.linalg.norm(base_error) + 1e-10)
        w_com = 0.25 / (np.linalg.norm(com_error) + 1e-10)
        w_f1 = 0.25 / (np.linalg.norm(f1_error) + 1e-10)
        w_f2 = 0.25 / (np.linalg.norm(f2_error) + 1e-10)
        
        errors = np.concatenate([
            w_base * base_error,
            w_com * com_error,
            w_f1 * f1_error,
            w_f2 * f2_error
        ])
        
        jacobian_pinv = np.linalg.pinv(jacobian, rcond=1e-6)
        dq_all = alpha * jacobian_pinv @ errors
        dq_all = np.clip(dq_all, -0.2, 0.2)
        
        q[0:3] += dq_all[0:3]
        for i in range(6):
            q[6 + i] = np.clip(q[6 + i] + dq_all[3 + i], 
                               model.jnt_range[i, 0], model.jnt_range[i, 1])
    
    print(f"Inverse magnitude weighting: Base Z={q[2]:.4f} (error={total_error:.6f}) after {iteration+1} iters")
