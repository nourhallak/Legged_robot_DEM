#!/usr/bin/env python3
"""
Full walking simulation test: 400 steps with 5mm stride
Records foot positions and checks for sliding
"""

import mujoco
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

# Load trajectories
base_traj = np.load('base_trajectory.npy')
com_traj = np.load('com_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

# Get site IDs
com_site_id = model.site(name='com_site').id
foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id

# IK solver
def compute_ik_solution(model, data, base_target, com_target, foot1_target, foot2_target, max_iterations=50, tolerance=0.002):
    qpos = data.qpos.copy()
    alpha = 0.08
    epsilon = 1e-6
    
    for iteration in range(max_iterations):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        
        base_pos = qpos[0:3].copy()
        com_pos = data.site_xpos[com_site_id].copy()
        foot1_pos = data.site_xpos[foot1_site_id].copy()
        foot2_pos = data.site_xpos[foot2_site_id].copy()
        
        base_error = base_target - base_pos
        com_error = com_target - com_pos
        foot1_error = foot1_target - foot1_pos
        foot2_error = foot2_target - foot2_pos
        
        total_error = (np.linalg.norm(base_error) + np.linalg.norm(com_error) + 
                      np.linalg.norm(foot1_error) + np.linalg.norm(foot2_error))
        
        if total_error < tolerance:
            return qpos, iteration
        
        jacobian = np.zeros((12, 9))
        jacobian[0:3, 0:3] = np.eye(3)
        
        for j in range(6):
            qpos_plus = qpos.copy()
            qpos_plus[6 + j] += epsilon
            data.qpos[:] = qpos_plus
            mujoco.mj_forward(model, data)
            
            jacobian[3:6, 3 + j] = (data.site_xpos[com_site_id] - com_pos) / epsilon
            jacobian[6:9, 3 + j] = (data.site_xpos[foot1_site_id] - foot1_pos) / epsilon
            jacobian[9:12, 3 + j] = (data.site_xpos[foot2_site_id] - foot2_pos) / epsilon
        
        try:
            jacobian_pinv = np.linalg.pinv(jacobian, rcond=1e-6)
        except:
            return qpos, max_iterations
        
        errors = np.concatenate([base_error, com_error, foot1_error, foot2_error])
        dq_all = alpha * jacobian_pinv @ errors
        
        if np.any(~np.isfinite(dq_all)):
            return qpos, max_iterations
        
        dq_all = np.clip(dq_all, -0.2, 0.2)
        qpos[0:3] += dq_all[0:3]
        
        for i in range(6):
            qpos[6 + i] = qpos[6 + i] + dq_all[3 + i]
            qpos[6 + i] = np.clip(qpos[6 + i], model.jnt_range[i, 0], model.jnt_range[i, 1])
        
        qpos[3:7] = [1, 0, 0, 0]
    
    return qpos, max_iterations

# Run simulation
print("\n=== FULL WALKING SIMULATION (400 STEPS, 5MM STRIDE) ===\n")

qpos = data.qpos.copy()
foot1_x_errors = []
foot2_x_errors = []
foot1_z_errors = []
foot2_z_errors = []

for step in range(len(base_traj)):
    # Solve IK
    qpos, iters = compute_ik_solution(
        model, data,
        base_traj[step],
        com_traj[step],
        foot1_traj[step],
        foot2_traj[step]
    )
    
    # Get actual positions
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    foot1_pos = data.site_xpos[foot1_site_id]
    foot2_pos = data.site_xpos[foot2_site_id]
    
    # Compute errors
    foot1_x_err = (foot1_traj[step, 0] - foot1_pos[0]) * 1000
    foot2_x_err = (foot2_traj[step, 0] - foot2_pos[0]) * 1000
    foot1_z_err = (foot1_traj[step, 2] - foot1_pos[2]) * 1000
    foot2_z_err = (foot2_traj[step, 2] - foot2_pos[2]) * 1000
    
    foot1_x_errors.append(foot1_x_err)
    foot2_x_errors.append(foot2_x_err)
    foot1_z_errors.append(foot1_z_err)
    foot2_z_errors.append(foot2_z_err)

foot1_x_errors = np.array(foot1_x_errors)
foot2_x_errors = np.array(foot2_x_errors)
foot1_z_errors = np.array(foot1_z_errors)
foot2_z_errors = np.array(foot2_z_errors)

print("FOOT POSITION ERRORS (mm):")
print("-" * 60)
print(f"Foot 1 X:")
print(f"  Mean: {foot1_x_errors.mean():.2f}mm, Std: {foot1_x_errors.std():.2f}mm")
print(f"  Min: {foot1_x_errors.min():.2f}mm, Max: {foot1_x_errors.max():.2f}mm")
print()
print(f"Foot 2 X:")
print(f"  Mean: {foot2_x_errors.mean():.2f}mm, Std: {foot2_x_errors.std():.2f}mm")
print(f"  Min: {foot2_x_errors.min():.2f}mm, Max: {foot2_x_errors.max():.2f}mm")
print()
print(f"Foot 1 Z:")
print(f"  Mean: {foot1_z_errors.mean():.2f}mm, Std: {foot1_z_errors.std():.2f}mm")
print(f"  Min: {foot1_z_errors.min():.2f}mm, Max: {foot1_z_errors.max():.2f}mm")
print()
print(f"Foot 2 Z:")
print(f"  Mean: {foot2_z_errors.mean():.2f}mm, Std: {foot2_z_errors.std():.2f}mm")
print(f"  Min: {foot2_z_errors.min():.2f}mm, Max: {foot2_z_errors.max():.2f}mm")
print()

print("=" * 60)
print("WALKING QUALITY ASSESSMENT:")
print("=" * 60)

max_x_slip = max(abs(foot1_x_errors).max(), abs(foot2_x_errors).max())
mean_x_slip = (abs(foot1_x_errors).mean() + abs(foot2_x_errors).mean()) / 2

if max_x_slip < 5:
    print("✓ EXCELLENT: Feet have minimal sliding (< 5mm)")
elif max_x_slip < 10:
    print("✓ GOOD: Feet sliding acceptable (< 10mm)")
elif max_x_slip < 20:
    print("⚠ FAIR: Noticeable foot sliding (< 20mm)")
else:
    print("✗ POOR: Significant foot sliding (> 20mm)")

print()
print("Robot should walk smoothly for 400 steps (8.0cm total distance)")
print(f"with minimal sliding (average {mean_x_slip:.2f}mm per foot)")
