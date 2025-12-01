#!/usr/bin/env python3
"""Debug IK solver to see why base Z isn't updating"""
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

def compute_ik_debug(qpos, base_t, com_t, f1_t, f2_t, step_num, max_iters=50):
    """IK with debug output for first few iterations"""
    
    for iteration in range(max_iters):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        
        base_pos = qpos[0:3].copy()
        com_pos = data.site_xpos[com_id].copy()
        f1_pos = data.site_xpos[f1_id].copy()
        f2_pos = data.site_xpos[f2_id].copy()
        
        base_error = base_t - base_pos
        com_error = com_t - com_pos
        f1_error = f1_t - f1_pos
        f2_error = f2_t - f2_pos
        
        total_error = (np.linalg.norm(base_error) + np.linalg.norm(com_error) + 
                      np.linalg.norm(f1_error) + np.linalg.norm(f2_error))
        
        if iteration in [0, 1, 9, 49]:
            print(f"\nIteration {iteration}: Total error = {total_error:.6f}")
            print(f"  Base Z: actual={qpos[2]:.4f}, target={base_t[2]:.4f}, error={base_error[2]:.4f}")
        
        if total_error < 0.002:
            print(f"\nConverged at iteration {iteration}")
            break
        
        jacobian = np.zeros((12, 9))
        jacobian[0:3, 0:3] = np.eye(3)
        
        for j in range(6):
            qpos_plus = qpos.copy()
            qpos_plus[6 + j] += 1e-6
            
            data.qpos[:] = qpos_plus
            mujoco.mj_forward(model, data)
            
            pos_plus_com = data.site_xpos[com_id].copy()
            pos_plus_foot1 = data.site_xpos[f1_id].copy()
            pos_plus_foot2 = data.site_xpos[f2_id].copy()
            
            jacobian[3:6, 3 + j] = (pos_plus_com - com_pos) / 1e-6
            jacobian[6:9, 3 + j] = (pos_plus_foot1 - f1_pos) / 1e-6
            jacobian[9:12, 3 + j] = (pos_plus_foot2 - f2_pos) / 1e-6
        
        errors = np.concatenate([base_error, com_error, f1_error, f2_error])
        
        try:
            jacobian_pinv = np.linalg.pinv(jacobian, rcond=1e-6)
        except:
            print(f"Pinv failed")
            break
        
        dq_all = 0.08 * jacobian_pinv @ errors
        dq_all = np.clip(dq_all, -0.2, 0.2)
        
        qpos[0:3] += dq_all[0:3]
        
        for i in range(6):
            qpos[6 + i] = np.clip(qpos[6 + i] + dq_all[3 + i], 
                                 model.jnt_range[i, 0], model.jnt_range[i, 1])
        
        qpos[3:7] = [1, 0, 0, 0]
    
    print(f"Final error: {total_error:.6f}")
    print(f"Final base Z: {qpos[2]:.4f} (target {base_t[2]:.4f})")

# Test step 100 (should have base at 0.2m for stance)
print("Testing IK solver on step 100 (foot1 stance phase)")
q = data.qpos.copy()
compute_ik_debug(q, base_traj[100], com_traj[100], foot1_traj[100], foot2_traj[100], 100)
