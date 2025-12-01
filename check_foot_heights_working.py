#!/usr/bin/env python3
"""Test: IK without error normalization (like ik_simulation.py)"""
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

def compute_ik_no_normalize(qpos, base_t, com_t, f1_t, f2_t, max_iters=50):
    """IK without error normalization (like ik_simulation.py)"""
    
    for iter_count in range(max_iters):
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
        
        if total_error < 0.002:
            break
        
        J = np.zeros((12, 9))
        J[0:3, 0:3] = np.eye(3)
        
        for j in range(6):
            qp = qpos.copy()
            qp[6+j] += 1e-6
            data.qpos[:] = qp
            mujoco.mj_forward(model, data)
            
            J[3:6, 3 + j] = (data.site_xpos[com_id] - com_pos) / 1e-6
            J[6:9, 3 + j] = (data.site_xpos[f1_id] - f1_pos) / 1e-6
            J[9:12, 3 + j] = (data.site_xpos[f2_id] - f2_pos) / 1e-6
        
        # NO normalization - just use raw errors
        errors = np.concatenate([base_error, com_error, f1_error, f2_error])
        
        J_pinv = np.linalg.pinv(J, rcond=1e-6)
        dq = 0.08 * J_pinv @ errors
        dq = np.clip(dq, -0.2, 0.2)
        
        qpos[0:3] += dq[0:3]
        for i in range(6):
            qpos[6+i] = np.clip(qpos[6+i] + dq[3+i], model.jnt_range[i,0], model.jnt_range[i,1])
        qpos[3:7] = [1,0,0,0]
    
    return qpos

print("=== FOOT HEIGHT CHECK (95-5 smoothing, NO error normalization) ===\n")
print("Step | Base Z | Foot1 Z (mm) | Foot2 Z (mm)")
print("-" * 55)

q = data.qpos.copy()
prev_q = q.copy()

for step in range(0, len(base_traj), 100):
    q_before = q.copy()
    q = compute_ik_no_normalize(q, base_traj[step], com_traj[step], foot1_traj[step], foot2_traj[step])
    
    # Apply 95-5 smoothing
    q_smooth = 0.95 * q + 0.05 * prev_q
    prev_q = q_smooth.copy()
    
    # Get foot positions
    data.qpos[:] = q_smooth
    mujoco.mj_forward(model, data)
    
    base_z = q_smooth[2]
    f1_z = data.site_xpos[f1_id, 2] * 1000
    f2_z = data.site_xpos[f2_id, 2] * 1000
    
    status = f"base={base_z:.4f}"
    print(f"{step:3d} | {q_smooth[2]:.4f} | {f1_z:11.1f} | {f2_z:11.1f} | {status}")

print(f"\nExpected ranges: Foot Z = 210-225mm (ground + 15mm swing)")
