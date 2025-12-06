#!/usr/bin/env python3
"""Check robot model structure"""

import numpy as np
import mujoco
import re
from pathlib import Path

def load_model_with_assets(xml_path="legged_robot_ik.xml"):
    xml_file = Path(xml_path)
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
print(f"nq (DOF): {model.nq}")
print(f"nu (Actuators): {model.nu}")
print(f"njnt (Joints): {model.njnt}")
print()

print("Joints:")
for i in range(model.njnt):
    jnt_id = i
    jnt_type = model.jnt_type[i]
    jnt_name = model.names[model.name_jntadr[i]:].split(b'\x00')[0].decode()
    print(f"  {i}: {jnt_name} (type {jnt_type})")

print("\nActuators:")
for i in range(model.nu):
    act_name = model.names[model.name_actuatoradr[i]:].split(b'\x00')[0].decode()
    act_trnid = model.actuator_trnid[i]
    print(f"  {i}: {act_name} (transmits to joint {act_trnid[0]})")

print("\nqpos shape:", model.nq)
print("ctrl shape:", model.nu)
