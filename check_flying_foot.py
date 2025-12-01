#!/usr/bin/env python3
"""Quick check: sample 50 steps to see which foot is flying"""
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

def compute_ik(q, base_t, com_t, f1_t, f2_t, iters=20):
    for _ in range(iters):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        
        base_pos = q[0:3].copy()
        com_pos = data.site_xpos[com_id].copy()
        f1_pos = data.site_xpos[f1_id].copy()
        f2_pos = data.site_xpos[f2_id].copy()
        
        errs = np.concatenate([base_t-base_pos, com_t-com_pos, f1_t-f1_pos, f2_t-f2_pos])
        if np.linalg.norm(errs) < 0.002:
            break
        
        J = np.zeros((12, 9))
        J[0:3, 0:3] = np.eye(3)
        
        for j in range(6):
            qp = q.copy()
            qp[6+j] += 1e-6
            data.qpos[:] = qp
            mujoco.mj_forward(model, data)
            J[:, 3+j] = (np.concatenate([qp[0:3]-base_t, data.site_xpos[com_id]-com_t, 
                                        data.site_xpos[f1_id]-f1_t, data.site_xpos[f2_id]-f2_t]) - errs) / 1e-6
        
        dq = 0.08 * np.linalg.pinv(J, rcond=1e-6) @ errs
        dq = np.clip(dq, -0.2, 0.2)
        q[0:3] += dq[0:3]
        for i in range(6):
            q[6+i] = np.clip(q[6+i] + dq[3+i], model.jnt_range[i,0], model.jnt_range[i,1])
        q[3:7] = [1,0,0,0]
    return q

q = data.qpos.copy()
print("\n=== FOOT HEIGHT CHECK (Sample Steps) ===\n")
print("Step | Foot1 Z (mm) | Foot2 Z (mm) | Which is flying?")
print("-" * 55)

for step in [50, 100, 150, 200, 250, 300, 350]:
    q = compute_ik(q, base_traj[step], com_traj[step], foot1_traj[step], foot2_traj[step])
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    
    base_z = q[2]
    f1_z = data.site_xpos[f1_id, 2]
    f2_z = data.site_xpos[f2_id, 2]
    f1_target_z = foot1_traj[step, 2]
    f2_target_z = foot2_traj[step, 2]
    
    f1_error = (f1_target_z - f1_z) * 1000
    f2_error = (f2_target_z - f2_z) * 1000
    
    flying = "Foot1" if abs(f1_error) > 5 else ("Foot2" if abs(f2_error) > 5 else "Both good")
    
    print(f"{step:3d} | Base Z: {base_z:.4f}m | Foot1 Z: {f1_z*1000:8.1f}mm (err {f1_error:+.1f}mm) | Foot2 Z: {f2_z*1000:8.1f}mm (err {f2_error:+.1f}mm) | {flying}")
