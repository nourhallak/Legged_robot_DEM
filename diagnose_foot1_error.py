#!/usr/bin/env python3
"""Debug: Trace foot1 error through IK iterations"""
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

def compute_ik_debug(model, data, base_target, com_target, foot1_target, foot2_target, max_iterations=50):
    """IK with debug output"""
    
    qpos = data.qpos.copy()
    
    for iteration in range(max_iterations):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        
        base_pos = qpos[0:3].copy()
        com_pos = data.site_xpos[com_id].copy()
        foot1_pos = data.site_xpos[f1_id].copy()
        foot2_pos = data.site_xpos[f2_id].copy()
        
        base_error = base_target - base_pos
        com_error = com_target - com_pos
        foot1_error = foot1_target - foot1_pos
        foot2_error = foot2_target - foot2_pos
        
        total_error = (np.linalg.norm(base_error) + np.linalg.norm(com_error) + 
                      np.linalg.norm(foot1_error) + np.linalg.norm(foot2_error))
        
        if iteration in [0, 1, 10, 49]:
            print(f"Iter {iteration:2d}: Total error={total_error:.6f}")
            print(f"  Base error: {np.linalg.norm(base_error):.6f}, Foot1 error: {np.linalg.norm(foot1_error):.6f}")
            print(f"  Foot1 target: {foot1_target[2]:.4f}, actual: {foot1_pos[2]:.4f}, diff: {foot1_error[2]:.6f}")
        
        if total_error < 0.002:
            print(f"Converged at iteration {iteration}")
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
            jacobian[6:9, 3 + j] = (pos_plus_foot1 - foot1_pos) / 1e-6
            jacobian[9:12, 3 + j] = (pos_plus_foot2 - foot2_pos) / 1e-6
        
        errors = np.concatenate([base_error, com_error, foot1_error, foot2_error])
        jacobian_pinv = np.linalg.pinv(jacobian, rcond=1e-6)
        dq_all = 0.08 * jacobian_pinv @ errors
        dq_all = np.clip(dq_all, -0.2, 0.2)
        
        qpos[0:3] += dq_all[0:3]
        for i in range(6):
            qpos[6+i] = np.clip(qpos[6+i] + dq_all[3+i], model.jnt_range[i,0], model.jnt_range[i,1])
        qpos[3:7] = [1, 0, 0, 0]
    
    return qpos

print("=== IK Debug: Step 0 ===\n")

qpos_solution = compute_ik_debug(
    model, data,
    base_traj[0], com_traj[0], foot1_traj[0], foot2_traj[0],
    max_iterations=50
)

print(f"\nFinal foot1 position: {data.site_xpos[f1_id][2]:.4f}")
print(f"Target: {foot1_traj[0][2]:.4f}")
print(f"Error: {foot1_traj[0][2] - data.site_xpos[f1_id][2]:.6f}")
