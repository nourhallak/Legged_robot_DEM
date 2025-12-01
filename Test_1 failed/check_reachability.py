#!/usr/bin/env python3
"""
Check robot reachability: Can the robot actually achieve the target configuration?
Test by setting joint angles to neutral stance and checking if feet reach ground.
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

# Load trajectories to see what we're asking for
base_traj = np.load('base_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

# Get site IDs
com_site_id = model.site(name='com_site').id
foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id

print("\n=== ROBOT REACHABILITY CHECK ===\n")

# Test 1: Rest pose
print("Test 1: REST POSE (qpos0)")
data.qpos[:] = model.qpos0
mujoco.mj_forward(model, data)
print(f"  Base: {data.qpos[0:3]}")
print(f"  COM:  {data.site_xpos[com_site_id]}")
print(f"  Foot1: {data.site_xpos[foot1_site_id]}")
print(f"  Foot2: {data.site_xpos[foot2_site_id]}")

# Test 2: Standing pose (from trajectory first frame)
print("\nTest 2: STANDING POSE (step 0)")
qpos = np.zeros(13)
qpos[0:3] = base_traj[0]
qpos[3:7] = [1, 0, 0, 0]
# Keep joint angles at 0 (neutral)
data.qpos[:] = qpos
mujoco.mj_forward(model, data)
print(f"  Base target: {base_traj[0]}")
print(f"  Base actual: {data.qpos[0:3]}")
print(f"  COM target: {[0, 0, 0.215]} (approx)")
print(f"  COM actual: {data.site_xpos[com_site_id]}")
print(f"  Foot1 target: {foot1_traj[0]}")
print(f"  Foot1 actual: {data.site_xpos[foot1_site_id]}")
print(f"  Foot2 target: {foot2_traj[0]}")
print(f"  Foot2 actual: {data.site_xpos[foot2_site_id]}")

foot1_z_error = (foot1_traj[0, 2] - data.site_xpos[foot1_site_id, 2]) * 1000
foot2_z_error = (foot2_traj[0, 2] - data.site_xpos[foot2_site_id, 2]) * 1000

print(f"\n  Foot1 Z error: {foot1_z_error:.2f}mm (need to bend knees ~{foot1_z_error/5:.1f}° if {foot1_z_error:.0f}mm)")
print(f"  Foot2 Z error: {foot2_z_error:.2f}mm")

# Test 3: Adjust knee angles to reach ground
print("\nTest 3: ADJUSTED KNEE ANGLES (trying to reach ground)")
qpos = np.zeros(13)
qpos[0:3] = base_traj[0]
qpos[3:7] = [1, 0, 0, 0]

# Bend knees (joint indices 7 and 10 for left and right knees)
# Negative knee angle = bending forward
qpos[7] = -0.5   # Left knee
qpos[10] = -0.5  # Right knee

data.qpos[:] = qpos
mujoco.mj_forward(model, data)

foot1_z = data.site_xpos[foot1_site_id, 2]
foot2_z = data.site_xpos[foot2_site_id, 2]

print(f"  Knees bent to -0.5 rad:")
print(f"  Foot1 Z: {foot1_z:.6f} m (target: {foot1_traj[0, 2]:.6f} m, error: {(foot1_traj[0, 2] - foot1_z)*1000:.2f}mm)")
print(f"  Foot2 Z: {foot2_z:.6f} m (target: {foot2_traj[0, 2]:.6f} m, error: {(foot2_traj[0, 2] - foot2_z)*1000:.2f}mm)")

print(f"\n=== CONCLUSION ===")
if abs(foot1_z - foot1_traj[0, 2]) < 0.005 and abs(foot2_z - foot2_traj[0, 2]) < 0.005:
    print("✓ Robot can reach target ground positions with small knee bend")
else:
    print("✗ Robot struggling to reach targets - may need base height adjustment")
    print(f"\nNote: Target base Z = {base_traj[0, 2]:.6f} m")
    print(f"      Ground level = 0.21 m")
    print(f"      Difference = {(base_traj[0, 2] - 0.21)*1000:.2f} mm")
