"""Simple test to understand forward kinematics"""
import mujoco
import numpy as np
import os
import re
from pathlib import Path

def load_model_with_assets():
    xml_path = Path(__file__).parent / "legged_robot_ik.xml"
    
    with open(xml_path, "r") as f:
        xml_content = f.read()
    
    asset_dir = Path(__file__).parent / "Legged_robot" / "meshes"
    
    for match in re.finditer(r'filename="([^"]*\.stl)"', xml_content):
        stl_file = match.group(1)
        full_path = asset_dir / stl_file
        
        if full_path.exists():
            xml_content = xml_content.replace(f'filename="{stl_file}"', f'filename="{full_path}"')
    
    return mujoco.MjModel.from_xml_string(xml_content)

model = load_model_with_assets()
data = mujoco.MjData(model)

print(f"Model has {model.nq} DOF")
mujoco.mj_forward(model, data)

com_site_id = model.site(name='com_site').id
foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id

print(f"\nWith qpos = {data.qpos}:")
print(f"  COM: {data.site_xpos[com_site_id]}")
print(f"  Foot1: {data.site_xpos[foot1_site_id]}")
print(f"  Foot2: {data.site_xpos[foot2_site_id]}")

# Try a different initial configuration
print("\n\nTrying with qpos adjustments...")
data.qpos[:] = [0, 0, 0.1, 1, 0, 0.5, 0.5, -0.5, -0.5, 0, 0, 0, 0]
mujoco.mj_forward(model, data)

print(f"\nWith qpos = {data.qpos[:7]}...:")
print(f"  COM: {data.site_xpos[com_site_id]}")
print(f"  Foot1: {data.site_xpos[foot1_site_id]}")
print(f"  Foot2: {data.site_xpos[foot2_site_id]}")

# Check that we can access sites properly
print(f"\nTotal sites in model: {model.nsite}")
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    print(f"  Site {i}: {name}")
