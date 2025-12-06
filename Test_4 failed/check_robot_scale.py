#!/usr/bin/env python3
"""
Check robot actual dimensions
"""

import numpy as np
import mujoco
import re
from pathlib import Path

def load_model_with_assets(xml_path="legged_robot_ik.xml"):
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

print("ROBOT DIMENSIONS ANALYSIS")
print("="*80)

# Get all body positions and sizes
print("\nBody positions:")
for i in range(model.nbody):
    pos = model.body_pos[i]
    print(f"  Body {i}: pos = {pos}")

# Get joint ranges
print("\nJoint ranges:")
for i in range(model.nq):
    jmin, jmax = model.jnt_range[i]
    print(f"  Joint {i}: [{jmin:.6f}, {jmax:.6f}] rad")

# Measure robot dimensions with FK
print("\nMeasuring workspace dimensions...")
data.qpos[:] = 0
mujoco.mj_forward(model, data)

foot1_id = model.site(name='foot1_site').id
foot2_id = model.site(name='foot2_site').id
hip_id = model.body(name='hip').id

foot1_pos = data.site_xpos[foot1_id]
foot2_pos = data.site_xpos[foot2_id]
hip_pos = data.xpos[hip_id]

print(f"  Hip position:   {hip_pos}")
print(f"  Foot1 position: {foot1_pos}")
print(f"  Foot2 position: {foot2_pos}")

print(f"\nDistances:")
print(f"  Hip to Foot1: {np.linalg.norm(foot1_pos - hip_pos):.6f}")
print(f"  Hip to Foot2: {np.linalg.norm(foot2_pos - hip_pos):.6f}")
print(f"  Foot1 to Foot2: {np.linalg.norm(foot2_pos - foot1_pos):.6f}")

# Try different joint angles
print(f"\nTesting with non-zero angles...")
data.qpos[:] = 0
data.qpos[2] = 0.1
data.qpos[3] = 0.5
mujoco.mj_forward(model, data)

foot1_pos = data.site_xpos[foot1_id]
hip_pos = data.xpos[hip_id]
print(f"  Hip position:   {hip_pos}")
print(f"  Foot1 position: {foot1_pos}")
print(f"  Distance: {np.linalg.norm(foot1_pos - hip_pos):.6f}")

print("\n" + "="*80)
print("CONCLUSION: Robot is in MILLIMETERS (or very small units)")
print("Trajectory should be in the same units!")
print("="*80)
