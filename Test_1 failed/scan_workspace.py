#!/usr/bin/env python3
"""Check: Can foot1 reach Z=0.21m with any joint configuration?"""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

f1_id = model.site(name='foot1_site').id

print("Scanning joint space for foot1 Z position\n")
print("Joint 0 (Hip) | Foot1 Z | Diff from 0.21")
print("-" * 45)

# Fix other joints at resting position
q = data.qpos.copy()

# Scan hip joint (joint 0, index 6)
for hip_angle in np.linspace(model.jnt_range[0, 0], model.jnt_range[0, 1], 11):
    q[6] = hip_angle
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    f1_z = data.site_xpos[f1_id, 2]
    print(f"{hip_angle:9.4f} | {f1_z:.6f} | {f1_z - 0.21:.6f}")

print("\n\nNow scan knee joint (joint 1, index 7):")
print("Joint 1 (Knee) | Foot1 Z | Diff from 0.21")
print("-" * 45)

q = data.qpos.copy()

for knee_angle in np.linspace(model.jnt_range[1, 0], model.jnt_range[1, 1], 11):
    q[7] = knee_angle
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    f1_z = data.site_xpos[f1_id, 2]
    print(f"{knee_angle:9.4f} | {f1_z:.6f} | {f1_z - 0.21:.6f}")

print("\n\nNow scan ankle joint (joint 2, index 8):")
print("Joint 2 (Ankle) | Foot1 Z | Diff from 0.21")
print("-" * 45)

q = data.qpos.copy()

for ankle_angle in np.linspace(model.jnt_range[2, 0], model.jnt_range[2, 1], 11):
    q[8] = ankle_angle
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    f1_z = data.site_xpos[f1_id, 2]
    print(f"{ankle_angle:9.4f} | {f1_z:.6f} | {f1_z - 0.21:.6f}")
