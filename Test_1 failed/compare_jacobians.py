#!/usr/bin/env python3
"""Debug: Compare Jacobian computation methods"""
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

qpos = data.qpos.copy()

base_t = base_traj[0]
com_t = com_traj[0]
f1_t = foot1_traj[0]
f2_t = foot2_traj[0]

# ITERATION 0
data.qpos[:] = qpos
mujoco.mj_forward(model, data)

base_pos = qpos[0:3].copy()
com_pos = data.site_xpos[com_id].copy()
f1_pos = data.site_xpos[f1_id].copy()
f2_pos = data.site_xpos[f2_id].copy()

errs = np.concatenate([base_t-base_pos, com_t-com_pos, f1_t-f1_pos, f2_t-f2_pos])
err_norm = np.linalg.norm(errs)

print("Error vector (original):")
print(f"  Total error norm: {err_norm:.6f}\n")

# Compute J exactly like check_foot_heights_fixed.py
J = np.zeros((12, 9))
J[0:3, 0:3] = np.eye(3)

for j in range(6):
    qp = qpos.copy()
    qp[6+j] += 1e-6
    data.qpos[:] = qp
    mujoco.mj_forward(model, data)
    
    # This is what check_foot_heights_fixed.py does
    perturbed_error_vector = np.concatenate([qp[0:3]-base_t, data.site_xpos[com_id]-com_t, 
                                           data.site_xpos[f1_id]-f1_t, data.site_xpos[f2_id]-f2_t])
    J[:, 3+j] = (perturbed_error_vector - errs) / 1e-6

print(f"Jacobian from check_foot_heights_fixed.py method:")
print(f"  Condition: {np.linalg.cond(J):.6e}")
print(f"  Min/Max: {J.min():.6e} / {J.max():.6e}\n")

# Now compute it the "standard" way (like trace_ik_convergence.py)
J_standard = np.zeros((12, 9))
J_standard[0:3, 0:3] = np.eye(3)

for j in range(6):
    qp = qpos.copy()
    qp[6+j] += 1e-6
    data.qpos[:] = qp
    mujoco.mj_forward(model, data)
    
    # Compute the CHANGE in positions
    J_standard[3:6, 3 + j] = (data.site_xpos[com_id] - com_pos) / 1e-6
    J_standard[6:9, 3 + j] = (data.site_xpos[f1_id] - f1_pos) / 1e-6
    J_standard[9:12, 3 + j] = (data.site_xpos[f2_id] - f2_pos) / 1e-6

print(f"Jacobian standard method (change in position):")
print(f"  Condition: {np.linalg.cond(J_standard):.6e}")
print(f"  Min/Max: {J_standard.min():.6e} / {J_standard.max():.6e}\n")

# Try solving with both
errs_normalized = errs / (err_norm + 1e-10)

J_pinv = np.linalg.pinv(J, rcond=1e-6)
dq1 = 0.08 * J_pinv @ errs_normalized
print(f"dq with check_foot_heights method (with normalized errors): {dq1[0:3]}")

J_standard_pinv = np.linalg.pinv(J_standard, rcond=1e-6)
dq2 = 0.08 * J_standard_pinv @ errs_normalized
print(f"dq with standard method (with normalized errors): {dq2[0:3]}\n")

# Also try the standard method with UN-normalized errors (what trace_ik_convergence.py does)
dq3 = 0.08 * J_standard_pinv @ errs
print(f"dq with standard method (with UN-normalized errors): {dq3[0:3]}")
