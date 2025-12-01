#!/usr/bin/env python3
"""
Check what X positions the feet can actually reach.
Sweep knee and ankle angles and see foot X position range.
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

foot1_id = model.site(name='foot1_site').id
foot2_id = model.site(name='foot2_site').id

print("\n=== FOOT X REACH ANALYSIS ===\n")

# Test Foot 1 (left leg)
print("FOOT 1 X POSITIONS at different joint angles:")
print("Knee angle | Ankle angle | Foot1 X")
print("---------  |  -----------  | -------")

foot1_x_values = []
best_x = None

for knee_angle in np.linspace(-2.0, 0.5, 10):
    for ankle_angle in np.linspace(-1.57, 1.57, 20):
        qpos = model.qpos0.copy()
        qpos[7] = knee_angle    # Left knee
        qpos[8] = ankle_angle   # Left ankle
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        
        foot1_x = data.site_xpos[foot1_id, 0]
        foot1_x_values.append(foot1_x)
        
        if best_x is None or foot1_x > best_x:
            best_x = foot1_x

foot1_x_min = min(foot1_x_values)
foot1_x_max = max(foot1_x_values)

print(f"Foot1 X range: {foot1_x_min:.6f} to {foot1_x_max:.6f} m")
print(f"  Can reach X=0.0? {foot1_x_min <= 0 <= foot1_x_max}")
print()

# Same for Foot 2
print("FOOT 2 X POSITIONS at different joint angles:")

foot2_x_values = []

for knee_angle in np.linspace(-2.0, 0.5, 10):
    for ankle_angle in np.linspace(-1.57, 1.57, 20):
        qpos = model.qpos0.copy()
        qpos[10] = knee_angle   # Right knee
        qpos[11] = ankle_angle  # Right ankle
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        
        foot2_x = data.site_xpos[foot2_id, 0]
        foot2_x_values.append(foot2_x)

foot2_x_min = min(foot2_x_values)
foot2_x_max = max(foot2_x_values)

print(f"Foot2 X range: {foot2_x_min:.6f} to {foot2_x_max:.6f} m")
print(f"  Can reach X=0.06? {foot2_x_min <= 0.06 <= foot2_x_max}")
print()

print("=== CONCLUSION ===")
print(f"Foot1 needs X=0.0: {'✓ REACHABLE' if foot1_x_min <= 0 <= foot1_x_max else '✗ UNREACHABLE'}")
print(f"Foot2 needs X=0.06: {'✓ REACHABLE' if foot2_x_min <= 0.06 <= foot2_x_max else '✗ UNREACHABLE'}")
print()
print("If unreachable: Suggest targets within reachable range:")
print(f"  Foot1: max X = {foot1_x_max:.6f}")
print(f"  Foot2: max X = {foot2_x_max:.6f}")
