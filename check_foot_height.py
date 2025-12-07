#!/usr/bin/env python3
"""
Quick diagnostic: Check foot heights vs sand height
"""
import mujoco as mj
import numpy as np

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

mj.mj_resetData(model, data)

# Set initial pose
data.qpos[model.joint("hip_link_2_1").id] = 0.0
data.qpos[model.joint("link_2_1_link_1_1").id] = -0.5
data.qpos[model.joint("link_1_1_foot_1").id] = 0.0
data.qpos[model.joint("hip_link_2_2").id] = 0.0
data.qpos[model.joint("link_2_2_link_1_2").id] = -0.5
data.qpos[model.joint("link_1_2_foot_2").id] = 0.0

mj.mj_forward(model, data)

hip_pos = data.body('hip').xpos
foot1_pos = data.body('foot_1').xpos
foot2_pos = data.body('foot_2').xpos

print("\n" + "="*80)
print("FOOT HEIGHT DIAGNOSTIC")
print("="*80)
print(f"\nHip position:     Z={hip_pos[2]:.6f}m")
print(f"Foot 1 position:  Z={foot1_pos[2]:.6f}m (diff from hip: {foot1_pos[2]-hip_pos[2]:+.6f}m)")
print(f"Foot 2 position:  Z={foot2_pos[2]:.6f}m (diff from hip: {foot2_pos[2]-hip_pos[2]:+.6f}m)")
print(f"\nSand particle height: Z=0.442m")
print(f"Robot hip height:     Z={hip_pos[2]:.3f}m")
print(f"Robot feet height:    Z={foot1_pos[2]:.3f}m")

if foot1_pos[2] >= 0.442 - 0.003:  # 0.003 is sand radius
    print("\n[!] PROBLEM: Feet are ABOVE sand height!")
    print("[!] Feet cannot contact sand particles to push them")
    print("[!] Need to LOWER the feet by extending knees downward")
else:
    print(f"\n[OK] Feet are below sand height by {0.442 - foot1_pos[2]:.6f}m")

print("\n" + "="*80)
print("SOLUTION: Knee joint needs NEGATIVE angles (flex more) to lower feet")
print("Current knee angle: -0.5 rad")
print("Need: More negative angle (like -1.0 to -1.2 rad) to push feet into sand")
print("="*80 + "\n")
