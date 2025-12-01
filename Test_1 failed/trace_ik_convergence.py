#!/usr/bin/env python3
"""Check if learning rate or targets matter"""
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

# Run IK for STEP 0 for 50 iterations
q = data.qpos.copy()

base_target = base_traj[0]
com_target = com_traj[0]
f1_target = foot1_traj[0]
f2_target = foot2_traj[0]

print("Step 0 IK convergence:")
print("Iter | Base Z | Error | dq[2]")
print("-" * 45)

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
    
    if iteration in [0, 1, 5, 10, 20, 49]:
        print(f"{iteration:3d} | {q[2]:.4f} | {total_error:.6f} | ", end="")
    
    if total_error < 0.002:
        if iteration in [0, 1, 5, 10, 20, 49]:
            print("CONVERGED")
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
    
    errors = np.concatenate([base_error, com_error, f1_error, f2_error])
    
    jacobian_pinv = np.linalg.pinv(jacobian, rcond=1e-6)
    dq_all = 0.08 * jacobian_pinv @ errors
    dq_all = np.clip(dq_all, -0.2, 0.2)
    
    if iteration in [0, 1, 5, 10, 20, 49]:
        print(f"{dq_all[2]:10.6f}")
    
    q[0:3] += dq_all[0:3]
    for i in range(6):
        q[6 + i] = np.clip(q[6 + i] + dq_all[3 + i], 
                           model.jnt_range[i, 0], model.jnt_range[i, 1])

print(f"\nFinal base Z: {q[2]:.6f} (target: 0.2)")
