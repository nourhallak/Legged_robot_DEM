"""Find what joint configuration places feet at first trajectory point"""
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

com_traj = np.load("com_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

# Target for first frame
com_target = com_traj[0]
foot1_target = foot1_traj[0]
foot2_target = foot2_traj[0]

print(f"First frame targets:")
print(f"  COM: {com_target}")
print(f"  Foot1: {foot1_target}")
print(f"  Foot2: {foot2_target}")

# Get site IDs
com_site_id = model.site(name='com_site').id
foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id

print(f"\nWith initial qpos [0, 0, 0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]:")
data.qpos[:] = [0, 0, 0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mujoco.mj_forward(model, data)
print(f"  COM: {data.site_xpos[com_site_id]}")
print(f"  Foot1: {data.site_xpos[foot1_site_id]}")
print(f"  Foot2: {data.site_xpos[foot2_site_id]}")

print(f"\nTrying different configs to get feet to z=0...")

# Try different z heights for the base
for z_base in [0.0, 0.02, 0.04, 0.06, 0.08]:
    data.qpos[:] = [0, 0, z_base, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mujoco.mj_forward(model, data)
    foot1_z = data.site_xpos[foot1_site_id, 2]
    foot2_z = data.site_xpos[foot2_site_id, 2]
    com_z = data.site_xpos[com_site_id, 2]
    print(f"  z_base={z_base}: feet_z={foot1_z:.4f},{foot2_z:.4f}, com_z={com_z:.4f}")

print(f"\nTrying with joint angles to lower the feet...")
# Try some leg configurations
configs = [
    [0, 0, 0.1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0],
    [0, 0, 0.1, 1, 0, 0.8, 0.8, 0, 0.8, 0.8, 0, 0, 0],
    [0, 0, 0.05, 1, 0, 0.5, 0.5, 0, 0.5, 0.5, 0, 0, 0],
]

for cfg in configs:
    data.qpos[:] = cfg
    try:
        mujoco.mj_forward(model, data)
        foot1_z = data.site_xpos[foot1_site_id, 2]
        foot2_z = data.site_xpos[foot2_site_id, 2]
        foot1_x = data.site_xpos[foot1_site_id, 0]
        foot2_x = data.site_xpos[foot2_site_id, 0]
        print(f"  qpos[3:7]={cfg[3:7]}: foot1=({foot1_x:.4f}, z={foot1_z:.4f}), foot2=({foot2_x:.4f}, z={foot2_z:.4f})")
    except:
        print(f"  qpos={cfg[:7]}: ERROR")
