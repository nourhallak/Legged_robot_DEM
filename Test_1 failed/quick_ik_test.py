#!/usr/bin/env python3
"""
Quick test: Run first 10 simulation steps and print foot position errors
"""

import mujoco
import numpy as np
import os
import re

# Load model
xml_path = 'legged_robot_ik.xml'
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Load trajectories
base_traj = np.load('base_trajectory.npy')
com_traj = np.load('com_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

# Get site IDs
try:
    com_site_id = model.site(name='com_site').id
    foot1_site_id = model.site(name='foot1_site').id
    foot2_site_id = model.site(name='foot2_site').id
except Exception as e:
    print(f"Error finding sites: {e}")
    exit(1)

print("\n=== TESTING FIRST 10 STEPS WITH UNIFORM 15MM HEIGHTS ===\n")

qpos = data.qpos.copy()
alpha = 0.08

for step in range(10):
    # Simple IK iteration (5 iterations per step)
    for _ in range(5):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        
        base_pos = qpos[0:3].copy()
        com_pos = data.site_xpos[com_site_id].copy()
        foot1_pos = data.site_xpos[foot1_site_id].copy()
        foot2_pos = data.site_xpos[foot2_site_id].copy()
        
        base_error = base_traj[step] - base_pos
        com_error = com_traj[step] - com_pos
        foot1_error = foot1_traj[step] - foot1_pos
        foot2_error = foot2_traj[step] - foot2_pos
        
        errors = np.concatenate([base_error, com_error, foot1_error, foot2_error])
        if np.linalg.norm(errors) < 0.002:
            break
        
        # Numerical Jacobian
        epsilon = 1e-6
        J = np.zeros((12, 9))
        J[0:3, 0:3] = np.eye(3)
        
        for j in range(6):
            qp = qpos.copy()
            qp[6 + j] += epsilon
            data.qpos[:] = qp
            mujoco.mj_forward(model, data)
            
            e_plus = np.concatenate([
                base_traj[step] - qp[0:3],
                com_traj[step] - data.site_xpos[com_site_id],
                foot1_traj[step] - data.site_xpos[foot1_site_id],
                foot2_traj[step] - data.site_xpos[foot2_site_id]
            ])
            
            J[:, 3 + j] = (e_plus - errors) / epsilon
        
        # Update
        try:
            dq = alpha * np.linalg.pinv(J, rcond=1e-6) @ errors
            qpos[0:3] += dq[0:3]
            for i in range(6):
                qpos[6 + i] += dq[3 + i]
                qpos[6 + i] = np.clip(qpos[6 + i], 
                                     model.jnt_range[i, 0], 
                                     model.jnt_range[i, 1])
            qpos[3:7] = [1, 0, 0, 0]
        except:
            pass
    
    # Check final error
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    foot1_slip_mm = (foot1_traj[step, 0] - data.site_xpos[foot1_site_id, 0]) * 1000
    foot2_slip_mm = (foot2_traj[step, 0] - data.site_xpos[foot2_site_id, 0]) * 1000
    foot1_z_mm = (foot1_traj[step, 2] - data.site_xpos[foot1_site_id, 2]) * 1000
    foot2_z_mm = (foot2_traj[step, 2] - data.site_xpos[foot2_site_id, 2]) * 1000
    
    phase = "Stance" if (step % 200) < 100 else "Swing"
    print(f"Step {step:2d} ({phase}): Foot1 X={foot1_slip_mm:6.2f}mm Z={foot1_z_mm:6.2f}mm  |  " +
          f"Foot2 X={foot2_slip_mm:6.2f}mm Z={foot2_z_mm:6.2f}mm")

print("\nIf X and Z errors are < 5mm, IK is working well with uniform heights.")
