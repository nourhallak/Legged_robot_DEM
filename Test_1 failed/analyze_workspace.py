#!/usr/bin/env python3
"""
Analyze robot workspace and create proper walking trajectories
"""
import numpy as np
import mujoco
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model with corrected paths
xml_path = os.path.join(script_dir, "legged_robot_ik.xml")
with open(xml_path, 'r') as f:
    xml_content = f.read()
xml_content = xml_content.replace('Legged_robot/meshes/', '../Legged_robot/meshes/')
temp_xml = os.path.join(script_dir, "temp.xml")
with open(temp_xml, 'w') as f:
    f.write(xml_content)

model = mujoco.MjModel.from_xml_path(temp_xml)
data = mujoco.MjData(model)

print("\n" + "="*80)
print("WORKSPACE ANALYSIS")
print("="*80)

# Sample workspace
print("\nSampling workspace...")
positions = []
hip_angles = np.linspace(-np.pi/2.5, 0, 10)
knee_angles = np.linspace(-np.pi/2.5, 0, 10)
ankle_angles = [0]  # Flat foot

for h in hip_angles:
    for k in knee_angles:
        for a in ankle_angles:
            data.qpos[3:6] = [h, k, a]
            mujoco.mj_forward(model, data)
            positions.append(data.site_xpos[0].copy())

positions = np.array(positions)

print(f"\nFootprint workspace (left foot):")
print(f"  X: {positions[:, 0].min():.1f} to {positions[:, 0].max():.1f} mm")
print(f"  Y: {positions[:, 1].min():.1f} to {positions[:, 1].max():.1f} mm")
print(f"  Z: {positions[:, 2].min():.1f} to {positions[:, 2].max():.1f} mm")

# Test with default pose
data.qpos[:] = 0
mujoco.mj_forward(model, data)
print(f"\nDefault pose:")
print(f"  Hip (body 1): {data.xpos[1]}")
print(f"  Left foot: {data.site_xpos[0]}")
print(f"  Right foot: {data.site_xpos[1]}")

os.remove(temp_xml)
