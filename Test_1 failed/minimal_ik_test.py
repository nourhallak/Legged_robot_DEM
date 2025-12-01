#!/usr/bin/env python3
"""Minimal IK test - trace through exactly one iteration to see Jacobian"""
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

# Start from rest pose
q = data.qpos.copy()

# STEP 0 targets
base_target = base_traj[0]  # [0.0, 0.0, 0.2]
com_target = com_traj[0]    # [0.0, 0.0, 0.2152]
f1_target = foot1_traj[0]   # [0.0e+00, -2.0e-04, 2.1e-01]
f2_target = foot2_traj[0]   # [2.5e-03, 2.0e-04, 2.1e-01]

print(f"Targets: base={base_target}, com={com_target}")
print(f"         foot1={f1_target}, foot2={f2_target}\n")

# One iteration of IK
data.qpos[:] = q
mujoco.mj_forward(model, data)

base_pos = q[0:3].copy()
com_pos = data.site_xpos[com_id].copy()
f1_pos = data.site_xpos[f1_id].copy()
f2_pos = data.site_xpos[f2_id].copy()

print(f"Initial: base={base_pos}, com={com_pos}")
print(f"         foot1={f1_pos}, foot2={f2_pos}\n")

base_error = base_target - base_pos
com_error = com_target - com_pos
f1_error = f1_target - f1_pos
f2_error = f2_target - f2_pos

print(f"Errors (before norm):")
print(f"  base_error: {base_error}")
print(f"  com_error:  {com_error}")
print(f"  f1_error:   {f1_error}")
print(f"  f2_error:   {f2_error}\n")

print(f"Error norms:")
print(f"  ||base_error||: {np.linalg.norm(base_error):.6f}")
print(f"  ||com_error||:  {np.linalg.norm(com_error):.6f}")
print(f"  ||f1_error||:   {np.linalg.norm(f1_error):.6f}")
print(f"  ||f2_error||:   {np.linalg.norm(f2_error):.6f}\n")

errors = np.concatenate([base_error, com_error, f1_error, f2_error])
print(f"Total error vector: {errors}")
print(f"Total error norm: {np.linalg.norm(errors):.6f}\n")

# Compute Jacobian
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

print(f"Jacobian shape: {jacobian.shape}")
print(f"Jacobian column norms (shows scale of each variable's effect):")
for i in range(9):
    col_norm = np.linalg.norm(jacobian[:, i])
    print(f"  Col {i}: {col_norm:.6e}")

print(f"\nJacobian row norms (shows scale of each error component):")
for i in range(12):
    row_norm = np.linalg.norm(jacobian[i, :])
    print(f"  Row {i}: {row_norm:.6e}")

print(f"\nJacobian condition number: {np.linalg.cond(jacobian):.6e}")
print(f"Jacobian min/max: {jacobian.min():.6e} / {jacobian.max():.6e}\n")

jacobian_pinv = np.linalg.pinv(jacobian, rcond=1e-6)
print(f"Pinv condition number: {np.linalg.cond(jacobian_pinv):.6e}")

dq_all = 0.08 * jacobian_pinv @ errors
print(f"\ndq update: {dq_all}")
print(f"dq[0:3] (base): {dq_all[0:3]}")
print(f"||dq||: {np.linalg.norm(dq_all):.6e}")

# What about with weighted errors?
print("\n\n=== WITH ERROR WEIGHTING (normalize each target's error) ===\n")

errors_weighted = np.concatenate([
    base_error / (np.linalg.norm(base_error) + 1e-10),
    com_error / (np.linalg.norm(com_error) + 1e-10),
    f1_error / (np.linalg.norm(f1_error) + 1e-10),
    f2_error / (np.linalg.norm(f2_error) + 1e-10),
])

print(f"Weighted error vector: {errors_weighted}")
print(f"Weighted error norm: {np.linalg.norm(errors_weighted):.6f}\n")

dq_weighted = 0.08 * jacobian_pinv @ errors_weighted
print(f"dq update (weighted): {dq_weighted}")
print(f"dq[0:3] (base, weighted): {dq_weighted[0:3]}")
print(f"||dq|| (weighted): {np.linalg.norm(dq_weighted):.6e}")
