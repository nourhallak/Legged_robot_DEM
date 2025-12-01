#!/usr/bin/env python3
"""Check Jacobian sign: position-based vs error-based"""
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

q = data.qpos.copy()

base_target = base_traj[0]
com_target = com_traj[0]
f1_target = foot1_traj[0]
f2_target = foot2_traj[0]

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

errors_current = np.concatenate([base_error, com_error, f1_error, f2_error])

print("=== METHOD 1: Jacobian of POSITIONS (trace_ik_convergence.py) ===\n")

J1 = np.zeros((12, 9))
J1[0:3, 0:3] = np.eye(3)

for j in range(6):
    qp = q.copy()
    qp[6+j] += 1e-6
    data.qpos[:] = qp
    mujoco.mj_forward(model, data)
    
    # Jacobian of POSITIONS: how much do positions change?
    J1[3:6, 3 + j] = (data.site_xpos[com_id] - com_pos) / 1e-6
    J1[6:9, 3 + j] = (data.site_xpos[f1_id] - f1_pos) / 1e-6
    J1[9:12, 3 + j] = (data.site_xpos[f2_id] - f2_pos) / 1e-6

print(f"J1 condition: {np.linalg.cond(J1):.6e}")
J1_pinv = np.linalg.pinv(J1, rcond=1e-6)
dq1 = 0.08 * J1_pinv @ errors_current
print(f"dq1[0:3] (base): {dq1[0:3]}")
print(f"This should move base upward\n")

print("=== METHOD 2: Jacobian of ERRORS (check_foot_heights_fixed bug) ===\n")

J2 = np.zeros((12, 9))
J2[0:3, 0:3] = np.eye(3)

for j in range(6):
    qp = q.copy()
    qp[6+j] += 1e-6
    data.qpos[:] = qp
    mujoco.mj_forward(model, data)
    
    # Get perturbed error vector
    base_error_pert = base_target - qp[0:3]
    com_error_pert = com_target - data.site_xpos[com_id]
    f1_error_pert = f1_target - data.site_xpos[f1_id]
    f2_error_pert = f2_target - data.site_xpos[f2_id]
    errors_pert = np.concatenate([base_error_pert, com_error_pert, f1_error_pert, f2_error_pert])
    
    # Jacobian of ERRORS: how much does error vector change?
    J2[:, 3 + j] = (errors_pert - errors_current) / 1e-6

print(f"J2 condition: {np.linalg.cond(J2):.6e}")
J2_pinv = np.linalg.pinv(J2, rcond=1e-6)
dq2 = 0.08 * J2_pinv @ errors_current
print(f"dq2[0:3] (base): {dq2[0:3]}")
print(f"This should ALSO move base upward (or maybe have wrong sign?)\n")

print("=== METHOD 3: With negative error Jacobian ===\n")
J3 = -J2  # Flip sign
J3_pinv = np.linalg.pinv(J3, rcond=1e-6)
dq3 = 0.08 * J3_pinv @ errors_current
print(f"dq3[0:3] (base): {dq3[0:3]}")
print(f"This is -J2, should move base correctly if J2 had wrong sign\n")

print("=== ANALYSIS ===")
print(f"J1 and J2 are mathematically: J2 = -J1")
print(f"So they should be negatives of each other...")
print(f"J1[:3, :3] (position J for base):\n{J1[:3, :3]}")
print(f"J2[:3, :3] (error J for base):\n{J2[:3, :3]}")
