#!/usr/bin/env python3
"""
Find what joint configurations place the feet at reasonable positions
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from ik_simulation import load_model_with_assets
import mujoco

print("Loading model...")
model = load_model_with_assets()
data = mujoco.MjData(model)

foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id

print("\n" + "="*80)
print("SWEEPING JOINT ANGLES TO FIND REACHABLE POSITIONS")
print("="*80)

# Try different joint configurations
print("\nScanning joint 1 (hip_link_2_1):")
for j1 in np.linspace(-1.57, 1.57, 10):
    data.qpos[:] = 0
    data.qpos[6 + 1] = j1  # Set joint 1
    mujoco.mj_forward(model, data)
    
    f1 = data.site_xpos[foot1_site_id]
    f2 = data.site_xpos[foot2_site_id]
    print(f"  J1={j1:6.2f}: Foot1 X={f1[0]:7.4f} Z={f1[2]:7.4f}  |  Foot2 X={f2[0]:7.4f} Z={f2[2]:7.4f}")

print("\nScanning joint 2 (link_2_1_link_1_1):")
for j2 in np.linspace(-2.09, 1.05, 10):
    data.qpos[:] = 0
    data.qpos[6 + 2] = j2  # Set joint 2
    mujoco.mj_forward(model, data)
    
    f1 = data.site_xpos[foot1_site_id]
    f2 = data.site_xpos[foot2_site_id]
    print(f"  J2={j2:6.2f}: Foot1 X={f1[0]:7.4f} Z={f1[2]:7.4f}  |  Foot2 X={f2[0]:7.4f} Z={f2[2]:7.4f}")

print("\nTrying to reach target foot1=(0, 0, 0.21), foot2=(0.0025, 0, 0.21)")
print("Testing manual configuration...")

# Try a reasonable standing pose
configs = [
    ([0, -1.3, 0.8, 0, -0.2, 0.15], "Config A"),
    ([0, -1.4, 0.9, 0, -0.15, 0.1], "Config B"),
    ([0, -1.5, 1.0, 0, -0.1, 0.05], "Config C"),
    ([0, -1.2, 0.7, 0, -0.2, 0.15], "Config D"),
]

for joints, name in configs:
    data.qpos[:] = 0
    for i, j in enumerate(joints):
        data.qpos[6 + i] = j
    mujoco.mj_forward(model, data)
    
    f1 = data.site_xpos[foot1_site_id]
    f2 = data.site_xpos[foot2_site_id]
    
    f1_err = np.linalg.norm(f1 - [0, 0, 0.21]) * 1000
    f2_err = np.linalg.norm(f2 - [0.0025, 0, 0.21]) * 1000
    
    print(f"\n{name}: {joints}")
    print(f"  Foot1: [{f1[0]:7.4f}, {f1[1]:7.4f}, {f1[2]:7.4f}] error={f1_err:6.1f}mm")
    print(f"  Foot2: [{f2[0]:7.4f}, {f2[1]:7.4f}, {f2[2]:7.4f}] error={f2_err:6.1f}mm")
