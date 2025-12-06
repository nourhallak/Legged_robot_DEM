#!/usr/bin/env python3
"""
Measure actual link lengths from MuJoCo model
"""

import numpy as np
import mujoco
import re
from pathlib import Path

def load_model_with_assets(xml_path="legged_robot_ik.xml"):
    """Load MuJoCo model with correct asset paths"""
    xml_file = Path(xml_path)
    if not xml_file.exists():
        xml_file = Path(".") / "legged_robot_ik.xml"
    
    with open(xml_file, "r") as f:
        xml_content = f.read()
    
    asset_dir = xml_file.parent / "Legged_robot" / "meshes"
    for match in re.finditer(r'filename="([^"]*\.stl)"', xml_content):
        stl_file = match.group(1)
        full_path = asset_dir / stl_file
        if full_path.exists():
            xml_content = xml_content.replace(f'filename="{stl_file}"', f'filename="{full_path}"')
    
    return mujoco.MjModel.from_xml_string(xml_content)

model = load_model_with_assets()
data = mujoco.MjData(model)

print("="*80)
print("MEASURING LEG GEOMETRY")
print("="*80)

# Zero configuration
data.qpos[:] = 0.0
data.qpos[2] = 0.42
mujoco.mj_forward(model, data)

hip_pos = data.xpos[model.body(name='hip').id].copy()
foot1_pos = data.site_xpos[model.site(name='foot1_site').id].copy()
foot2_pos = data.site_xpos[model.site(name='foot2_site').id].copy()

print(f"\nZero Configuration:")
print(f"  Hip position: {hip_pos}")
print(f"  Foot1 position: {foot1_pos}")
print(f"  Foot2 position: {foot2_pos}")
print(f"  Distance hip->foot1: {np.linalg.norm(foot1_pos - hip_pos)*1000:.2f} mm")
print(f"  Distance hip->foot2: {np.linalg.norm(foot2_pos - hip_pos)*1000:.2f} mm")

# Bend left leg
data.qpos[:] = 0.0
data.qpos[2] = 0.42
data.qpos[3] = np.pi/2  # Hip 90 degrees
data.qpos[4] = -np.pi/2  # Knee -90 degrees
data.qpos[5] = 0  # Ankle
mujoco.mj_forward(model, data)

foot1_bent = data.site_xpos[model.site(name='foot1_site').id].copy()
print(f"\nLeft leg bent (hip=90°, knee=-90°):")
print(f"  Foot1 position: {foot1_bent}")
print(f"  Distance hip->foot1: {np.linalg.norm(foot1_bent - hip_pos)*1000:.2f} mm")

# Stretch left leg
data.qpos[:] = 0.0
data.qpos[2] = 0.42
data.qpos[3] = 0  # Hip 0 degrees
data.qpos[4] = 0  # Knee 0 degrees (straight)
data.qpos[5] = 0  # Ankle
mujoco.mj_forward(model, data)

foot1_straight = data.site_xpos[model.site(name='foot1_site').id].copy()
hip_pos_straight = data.xpos[model.body(name='hip').id].copy()

print(f"\nLeft leg straight (hip=0°, knee=0°):")
print(f"  Hip position: {hip_pos_straight}")
print(f"  Foot1 position: {foot1_straight}")
print(f"  Total distance: {np.linalg.norm(foot1_straight - hip_pos_straight)*1000:.2f} mm")

# Try different knee angles to find link lengths
print(f"\nScanning knee angle to estimate link length:")
for knee_deg in [0, 30, 60, 90]:
    data.qpos[:] = 0.0
    data.qpos[2] = 0.42
    data.qpos[3] = 0
    data.qpos[4] = -np.radians(knee_deg)
    data.qpos[5] = 0
    mujoco.mj_forward(model, data)
    
    foot_pos = data.site_xpos[model.site(name='foot1_site').id].copy()
    dist = np.linalg.norm(foot_pos - hip_pos_straight)
    print(f"  Knee angle ->{knee_deg:3d}°: distance = {dist*1000:.2f} mm")

print("\n" + "="*80)
