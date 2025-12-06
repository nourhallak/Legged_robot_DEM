#!/usr/bin/env python3
"""
Analyze IK errors to understand convergence issues
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
data = mujoco.MjData(model)

# Load trajectories and solutions
base_traj = np.load("base_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")
ik_solutions = np.load("joint_solutions_ik.npy")

print("="*80)
print("IK ERROR ANALYSIS")
print("="*80)

# Analyze errors for key frames
errors_per_frame = []

for step in range(len(ik_solutions)):
    qpos = ik_solutions[step]
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    # Get actual positions
    base_id = model.body(name='hip').id
    foot1_id = model.site(name='foot1_site').id
    foot2_id = model.site(name='foot2_site').id
    
    base_actual = data.xpos[base_id]
    foot1_actual = data.site_xpos[foot1_id]
    foot2_actual = data.site_xpos[foot2_id]
    
    # Targets
    base_target = base_traj[step]
    foot1_target = foot1_traj[step]
    foot2_target = foot2_traj[step]
    
    # Errors
    base_err = np.linalg.norm(base_actual - base_target)
    foot1_err = np.linalg.norm(foot1_actual - foot1_target)
    foot2_err = np.linalg.norm(foot2_actual - foot2_target)
    
    total_err = base_err + foot1_err + foot2_err
    errors_per_frame.append(total_err)
    
    # Print problematic frames
    if step < 5 or total_err > 0.035:
        print(f"\nFrame {step}:")
        print(f"  Base error: {base_err:.5f}m ({base_err*1000:.2f}mm)")
        print(f"  Foot1 error: {foot1_err:.5f}m ({foot1_err*1000:.2f}mm)")
        print(f"  Foot2 error: {foot2_err:.5f}m ({foot2_err*1000:.2f}mm)")
        print(f"  Total: {total_err:.5f}m")

errors_per_frame = np.array(errors_per_frame)
print("\n" + "="*80)
print("STATISTICS")
print("="*80)
print(f"Mean error: {errors_per_frame.mean():.5f}m ({errors_per_frame.mean()*1000:.2f}mm)")
print(f"Max error: {errors_per_frame.max():.5f}m ({errors_per_frame.max()*1000:.2f}mm)")
print(f"Min error: {errors_per_frame.min():.5f}m ({errors_per_frame.min()*1000:.2f}mm)")
print(f"Std dev: {errors_per_frame.std():.5f}m ({errors_per_frame.std()*1000:.2f}mm)")

# Count high-error frames
high_error_frames = np.sum(errors_per_frame > 0.035)
print(f"Frames with error > 35mm: {high_error_frames}")

# Identify worst frame
worst_idx = np.argmax(errors_per_frame)
print(f"\nWorst frame: {worst_idx} with error {errors_per_frame[worst_idx]:.5f}m")
print(f"  Target foot1: {foot1_traj[worst_idx]}")
print(f"  Target foot2: {foot2_traj[worst_idx]}")
