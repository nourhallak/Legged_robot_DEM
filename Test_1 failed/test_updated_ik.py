#!/usr/bin/env python3
"""Quick test of updated ik_simulation.py logic"""
import sys
sys.path.insert(0, '.')

# Import from ik_simulation
import mujoco
import numpy as np

# Load and test the IK function directly
model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

base_traj = np.load('base_trajectory.npy')
com_traj = np.load('com_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

com_id = model.site(name='com_site').id
f1_id = model.site(name='foot1_site').id
f2_id = model.site(name='foot2_site').id

# Test the new IK function from ik_simulation.py
exec(open('ik_simulation.py').read(), globals())

print("=== Testing Updated IK (BASE + FEET only) ===\n")
print("Step | Base Z | Foot1 Z | Foot2 Z | Errors (mm)")
print("-" * 60)

prev_qpos = data.qpos.copy()
flying_foot_count = 0

for step_index in range(0, len(base_traj), 50):
    base_target = base_traj[step_index]
    com_target = com_traj[step_index]
    foot1_target = foot1_traj[step_index]
    foot2_target = foot2_traj[step_index]
    
    qpos_solution, success = compute_ik_solution(
        model, data,
        base_target, com_target, foot1_target, foot2_target,
        max_iterations=50, tolerance=0.002
    )
    
    # 95-5 smoothing
    smoothed_qpos = 0.95 * qpos_solution + 0.05 * prev_qpos
    
    # Update data
    data.qpos[:] = smoothed_qpos
    mujoco.mj_forward(model, data)
    
    # Get foot positions and errors
    f1_z = data.site_xpos[f1_id, 2] * 1000
    f2_z = data.site_xpos[f2_id, 2] * 1000
    target_f1 = foot1_target[2] * 1000
    target_f2 = foot2_target[2] * 1000
    f1_err = f1_z - target_f1
    f2_err = f2_z - target_f2
    
    print(f"{step_index:3d} | {smoothed_qpos[2]:6.4f} | {f1_z:7.1f} | {f2_z:7.1f} | F1:{f1_err:6.1f} F2:{f2_err:6.1f}")
    
    # Count flying feet
    if abs(f1_err) > 10:
        flying_foot_count += 1
    if abs(f2_err) > 10:
        flying_foot_count += 1
    
    # Save for next iteration
    prev_qpos = smoothed_qpos.copy()

print(f"\nTotal foot errors > 10mm: {flying_foot_count}")
print(f"Status: {'✓ FIXED' if flying_foot_count == 0 else '✗ STILL BROKEN'}")
