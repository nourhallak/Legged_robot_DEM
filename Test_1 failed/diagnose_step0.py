#!/usr/bin/env python3
"""Diagnose: Run IK step 0 with exact same setup as check_foot_heights_fixed.py"""
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

# EXACTLY replicate check_foot_heights_fixed.py setup
q = data.qpos.copy()
prev_q = q.copy()

# Step 0
base_target = base_traj[0]
com_target = com_traj[0]
f1_target = foot1_traj[0]
f2_target = foot2_traj[0]

print("Initial state:")
print(f"  q[0:3] = {q[0:3]}")
print(f"  Target: base={base_target}, com={com_target}")
print(f"  Target: foot1={f1_target}, foot2={f2_target}\n")

# Iteration 0 of IK
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

print("Iteration 0 errors:")
print(f"  base_error = {base_error}, norm = {np.linalg.norm(base_error):.6f}")
print(f"  com_error = {com_error}, norm = {np.linalg.norm(com_error):.6f}")
print(f"  f1_error = {f1_error}, norm = {np.linalg.norm(f1_error):.6f}")
print(f"  f2_error = {f2_error}, norm = {np.linalg.norm(f2_error):.6f}\n")

errors_original = np.concatenate([base_error, com_error, f1_error, f2_error])
err_norm = np.linalg.norm(errors_original)
print(f"Total error norm: {err_norm:.6f}")
print(f"Error vector: {errors_original}\n")

# Compute Jacobian
jacobian = np.zeros((12, 9))
jacobian[0:3, 0:3] = np.eye(3)

for j in range(6):
    qp = q.copy()
    qp[6+j] += 1e-6
    data.qpos[:] = qp
    mujoco.mj_forward(model, data)
    jacobian[:, 3+j] = (np.concatenate([qp[0:3]-base_target, data.site_xpos[com_id]-com_target, 
                                        data.site_xpos[f1_id]-f1_target, data.site_xpos[f2_id]-f2_target]) - errors_original) / 1e-6

print(f"Jacobian condition: {np.linalg.cond(jacobian):.6e}")
print(f"Jacobian min: {jacobian.min():.6e}, max: {jacobian.max():.6e}\n")

# Normalize errors like the fixed version does
errs_normalized = errors_original / (err_norm + 1e-10)

J_pinv = np.linalg.pinv(jacobian, rcond=1e-6)
print(f"Pinv condition: {np.linalg.cond(J_pinv):.6e}\n")

dq = 0.08 * J_pinv @ errs_normalized
print(f"dq with normalized errors: {dq}")
print(f"dq[0:3] (base): {dq[0:3]}")
print(f"||dq||: {np.linalg.norm(dq):.6e}\n")

# Also try without normalizing
dq_raw = 0.08 * J_pinv @ errors_original
print(f"dq with raw errors: {dq_raw}")
print(f"dq[0:3] (base): {dq_raw[0:3]}")
print(f"||dq||: {np.linalg.norm(dq_raw):.6e}")
