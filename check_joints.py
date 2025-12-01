"""Check joint ranges and movability"""
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

print("Joint configuration:")
print(f"Total DOF: {model.nq}")
print(f"Total joints: {model.njnt}")
print()

for i in range(model.njnt):
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jnt_type = model.jnt_type[i]
    
    # Free joint has no range entry
    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        print(f"Joint {i}: {jnt_name:20s} type=FREE (6 DOF)")
    else:
        jnt_range = model.jnt_range[i - 6] if i >= 6 else model.jnt_range[i]
        print(f"Joint {i}: {jnt_name:20s} type={jnt_type}, range=[{jnt_range[0]:.3f}, {jnt_range[1]:.3f}]")

print("\n\nTesting leg joint movements:")

# Test moving leg joint 1 (joint 6 in qpos)
print("\nMoving hip_1_joint (qpos[6]):")
for angle in [0, 0.3, 0.6, -0.3, -0.6]:
    data.qpos[:] = [0, 0, 0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    data.qpos[6] = angle
    mujoco.mj_forward(model, data)
    
    com_site_id = model.site(name='com_site').id
    foot1_site_id = model.site(name='foot1_site').id
    
    print(f"  qpos[6]={angle:+.1f}: foot1_z={data.site_xpos[foot1_site_id, 2]:.4f}, com_z={data.site_xpos[com_site_id, 2]:.4f}")

print("\nMoving knee_1_joint (qpos[7]):")
for angle in [0, 0.3, 0.6, -0.3, -0.6]:
    data.qpos[:] = [0, 0, 0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    data.qpos[7] = angle
    mujoco.mj_forward(model, data)
    
    foot1_site_id = model.site(name='foot1_site').id
    print(f"  qpos[7]={angle:+.1f}: foot1_x={data.site_xpos[foot1_site_id, 0]:.4f}, foot1_z={data.site_xpos[foot1_site_id, 2]:.4f}")
