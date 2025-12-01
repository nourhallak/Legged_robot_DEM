"""
Debug script to understand the IK problem better
"""
import mujoco
import numpy as np
import os
import re
from pathlib import Path

# Load model
def load_model_with_assets():
    xml_path = Path(__file__).parent / "legged_robot_ik.xml"
    
    with open(xml_path, "r") as f:
        xml_content = f.read()
    
    asset_dir = Path(__file__).parent / "Legged_robot" / "meshes"
    
    # Resolve file paths
    for match in re.finditer(r'filename="([^"]*\.stl)"', xml_content):
        stl_file = match.group(1)
        full_path = asset_dir / stl_file
        
        if full_path.exists():
            xml_content = xml_content.replace(f'filename="{stl_file}"', f'filename="{full_path}"')
    
    return mujoco.MjModel.from_xml_string(xml_content)

model = load_model_with_assets()
data = mujoco.MjData(model)

print(f"Model DOF (nq): {model.nq}")
print(f"Number of joints: {model.njnt}")
print(f"Number of actuators: {model.na}")
print(f"Joint range shape: {model.jnt_range.shape}")
print(f"\nJoint info:")

for i in range(model.njnt):
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jnt_type = model.jnt_type[i]
    jnt_range = model.jnt_range[i]
    print(f"  Joint {i}: {jnt_name}, type={jnt_type}, range={jnt_range}")

print(f"\nSite info:")
for i in range(model.nsite):
    site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    print(f"  Site {i}: {site_name}")

# Check initial state
print(f"\nInitial qpos: {data.qpos}")
print(f"Initial state DOF: {len(data.qpos)}")

# Compute forward kinematics
mujoco.mj_forward(model, data)
com_site_id = model.site(name='com_site').id
foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id

print(f"\nInitial site positions:")
print(f"  COM: {data.site_xpos[com_site_id]}")
print(f"  Foot1: {data.site_xpos[foot1_site_id]}")
print(f"  Foot2: {data.site_xpos[foot2_site_id]}")

# Test trajectory targets
com_traj = np.load("com_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

print(f"\nFirst trajectory target:")
print(f"  COM target: {com_traj[0]}")
print(f"  Foot1 target: {foot1_traj[0]}")
print(f"  Foot2 target: {foot2_traj[0]}")

print(f"\nLast trajectory target:")
print(f"  COM target: {com_traj[-1]}")
print(f"  Foot1 target: {foot1_traj[-1]}")
print(f"  Foot2 target: {foot2_traj[-1]}")

# Print jacobian rank check
print(f"\nJacobian analysis:")
print(f"  Model has {model.nq} DOF")
print(f"  We need to track 9 coordinates (3 sites × 3)")
print(f"  With 6 floating DOF + 6 actuated = 12 DOF total")
print(f"  Jacobian will be 9×12 (underdetermined, good)")
