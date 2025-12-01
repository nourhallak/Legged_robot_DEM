#!/usr/bin/env python3
"""
Diagnose IK issues by checking what the solver is producing vs what's expected
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from ik_simulation import compute_ik_solution, load_model_with_assets
import mujoco

print("Loading model and trajectories...")
model = load_model_with_assets()
data = mujoco.MjData(model)

# Load trajectories
base_traj = np.load('base_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')
com_traj = np.load('com_trajectory.npy')

num_steps = len(base_traj)
print(f"Loaded {num_steps} trajectory steps\n")

# Test a few key steps
test_steps = [0, 100, 200, 300, 399]

print("=" * 80)
print("IK DIAGNOSTIC: Checking if IK produces reasonable joint angles")
print("=" * 80)

for step in test_steps:
    print(f"\n{'='*80}")
    print(f"STEP {step}")
    print(f"{'='*80}")
    
    # Get trajectory targets
    base_target = base_traj[step]
    foot1_target = foot1_traj[step]
    foot2_target = foot2_traj[step]
    com_target = com_traj[step]
    
    print(f"Targets:")
    print(f"  Base:  X={base_target[0]:.5f}, Y={base_target[1]:.5f}, Z={base_target[2]:.5f}")
    print(f"  Foot1: X={foot1_target[0]:.5f}, Y={foot1_target[1]:.5f}, Z={foot1_target[2]:.5f}")
    print(f"  Foot2: X={foot2_target[0]:.5f}, Y={foot2_target[1]:.5f}, Z={foot2_target[2]:.5f}")
    
    # Solve IK
    qpos_solution, success = compute_ik_solution(
        model, data, base_target, com_target, foot1_target, foot2_target,
        max_iterations=50, tolerance=0.005
    )
    
    print(f"\nIK Result: {'SUCCESS' if success else 'CONVERGED' if not success else 'FAILED'}")
    print(f"Joint angles (radians):")
    print(f"  Base XYZ:    {qpos_solution[0:3]}")
    print(f"  Base Quat:   {qpos_solution[3:7]}")
    print(f"  Joints 0-5:  {qpos_solution[6:12]}")
    
    # Verify FK from this solution
    data.qpos[:] = qpos_solution
    mujoco.mj_forward(model, data)
    
    foot1_site_id = model.site(name='foot1_site').id
    foot2_site_id = model.site(name='foot2_site').id
    
    foot1_actual = data.site_xpos[foot1_site_id].copy()
    foot2_actual = data.site_xpos[foot2_site_id].copy()
    base_actual = qpos_solution[0:3]
    
    print(f"\nForward Kinematics Results:")
    print(f"  Base:  X={base_actual[0]:.5f}, Y={base_actual[1]:.5f}, Z={base_actual[2]:.5f}")
    print(f"  Foot1: X={foot1_actual[0]:.5f}, Y={foot1_actual[1]:.5f}, Z={foot1_actual[2]:.5f}")
    print(f"  Foot2: X={foot2_actual[0]:.5f}, Y={foot2_actual[1]:.5f}, Z={foot2_actual[2]:.5f}")
    
    print(f"\nErrors:")
    base_err = np.linalg.norm(base_target - base_actual)
    foot1_err = np.linalg.norm(foot1_target - foot1_actual)
    foot2_err = np.linalg.norm(foot2_target - foot2_actual)
    print(f"  Base error:  {base_err*1000:.2f} mm")
    print(f"  Foot1 error: {foot1_err*1000:.2f} mm")
    print(f"  Foot2 error: {foot2_err*1000:.2f} mm")
    
    # Check if joints are in reasonable ranges
    print(f"\nJoint Limits Check:")
    for i in range(6):
        q_val = qpos_solution[6 + i]
        q_min = model.jnt_range[i, 0]
        q_max = model.jnt_range[i, 1]
        in_range = q_min <= q_val <= q_max
        status = "OK" if in_range else "OUT OF RANGE"
        print(f"  Joint {i}: {q_val:7.4f} rad (limits: {q_min:7.4f} to {q_max:7.4f}) [{status}]")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
