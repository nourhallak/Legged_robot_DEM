#!/usr/bin/env python3
"""Check foot1 Z with different base positions"""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

f1_id = model.site(name='foot1_site').id

print("Foot1 Z vs Base Z (rest pose for legs):\n")
print("Base Z | Foot1 Z | Foot1 Z - Base Z")
print("-" * 40)

q = data.qpos.copy()

for base_z in np.linspace(0.1, 0.3, 11):
    q[2] = base_z
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    f1_z = data.site_xpos[f1_id, 2]
    print(f"{base_z:.3f} | {f1_z:.6f} | {f1_z - base_z:.6f}")

print("\n\nNow with knee bent to max:")
q[7] = 1.57  # Max knee extension
print("\nBase Z | Foot1 Z (knee=1.57) | Foot1 Z - Base Z")
print("-" * 45)

for base_z in np.linspace(0.1, 0.3, 11):
    q[2] = base_z
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    f1_z = data.site_xpos[f1_id, 2]
    print(f"{base_z:.3f} | {f1_z:14.6f} | {f1_z - base_z:.6f}")
