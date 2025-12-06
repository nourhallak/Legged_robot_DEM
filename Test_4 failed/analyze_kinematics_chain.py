#!/usr/bin/env python3
"""
Understand robot kinematics chain and joint effects
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
    
    # Update mesh paths
    asset_dir = xml_file.parent / "Legged_robot" / "meshes"
    
    for match in re.finditer(r'filename="([^"]*\.stl)"', xml_content):
        stl_file = match.group(1)
        full_path = asset_dir / stl_file
        
        if full_path.exists():
            xml_content = xml_content.replace(f'filename="{stl_file}"', f'filename="{full_path}"')
    
    return mujoco.MjModel.from_xml_string(xml_content)

print("="*80)
print("ROBOT KINEMATICS CHAIN ANALYSIS")
print("="*80)

# Load model
model = load_model_with_assets()
data = mujoco.MjData(model)

print(f"\nModel Info:")
print(f"  nq (DOF): {model.nq}")
print(f"  nu (actuators): {model.nu}")
print(f"  nbody: {model.nbody}")

print(f"\nJoints: {model.nq} total")
print(f"Joint types: {model.jnt_type}")

print(f"\nActuators: {model.nu} total")
for i in range(model.nu):
    act_trnid = model.actuator_trnid[i]
    print(f"  [{i}] -> qpos[{act_trnid[0]}]")

print(f"\nBodies: {model.nbody}")
print(f"\nSites: {model.nsite}")

# Test joint influence
print("\n" + "="*80)
print("TESTING JOINT INFLUENCE ON FEET")
print("="*80)

foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id

# Test each actuated joint
act_joints = [3, 4, 5, 6, 7, 8]

print(f"\nTesting effect of each joint on foot positions:")

for j_idx, j in enumerate(act_joints):
    # Zero configuration
    qpos_zero = np.zeros(model.nq)
    qpos_zero[2] = 0.42
    
    data.qpos[:] = qpos_zero
    mujoco.mj_forward(model, data)
    
    foot1_pos_zero = data.site_xpos[foot1_site_id].copy()
    foot2_pos_zero = data.site_xpos[foot2_site_id].copy()
    
    # Perturbed configuration (45 degrees)
    qpos_pert = qpos_zero.copy()
    qpos_pert[j] = np.pi / 4
    
    data.qpos[:] = qpos_pert
    mujoco.mj_forward(model, data)
    
    foot1_pos_pert = data.site_xpos[foot1_site_id].copy()
    foot2_pos_pert = data.site_xpos[foot2_site_id].copy()
    
    foot1_delta = np.linalg.norm(foot1_pos_pert - foot1_pos_zero)
    foot2_delta = np.linalg.norm(foot2_pos_pert - foot2_pos_zero)
    
    print(f"  Joint[{j}]:                             foot1_delta={foot1_delta:.6f}m, foot2_delta={foot2_delta:.6f}m")
    
    if foot1_delta > 1e-6 or foot2_delta > 1e-6:
        print(f"    Foot1: {foot1_pos_zero} -> {foot1_pos_pert} (delta={foot1_pos_pert - foot1_pos_zero})")
        print(f"    Foot2: {foot2_pos_zero} -> {foot2_pos_pert} (delta={foot2_pos_pert - foot2_pos_zero})")

print("\n" + "="*80)
