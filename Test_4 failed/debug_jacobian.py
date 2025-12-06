#!/usr/bin/env python3
"""
Debug Jacobian computation with detailed output
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

print("="*80)
print("JACOBIAN COMPUTATION DEBUG")
print("="*80)

model = load_model_with_assets()
data = mujoco.MjData(model)

foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id
base_body_id = model.body(name='hip').id

# Initial configuration
qpos_zero = np.zeros(model.nq)
qpos_zero[2] = 0.42

data.qpos[:] = qpos_zero
mujoco.mj_forward(model, data)

base_pos_zero = data.xpos[base_body_id].copy()
foot1_pos_zero = data.site_xpos[foot1_site_id].copy()
foot2_pos_zero = data.site_xpos[foot2_site_id].copy()

print(f"\nZero configuration foot positions:")
print(f"  Base:  {base_pos_zero}")
print(f"  Foot1: {foot1_pos_zero}")
print(f"  Foot2: {foot2_pos_zero}")

# Test joint 3
epsilon = 1e-3
j = 3

qpos_pert = qpos_zero.copy()
qpos_pert[j] += epsilon

print(f"\nPerturbation: joint {j} += {epsilon}")
print(f"  qpos_pert[{j}] = {qpos_pert[j]}")

data.qpos[:] = qpos_pert
mujoco.mj_forward(model, data)

base_pos_pert = data.xpos[base_body_id].copy()
foot1_pos_pert = data.site_xpos[foot1_site_id].copy()
foot2_pos_pert = data.site_xpos[foot2_site_id].copy()

print(f"\nPerturbed configuration foot positions:")
print(f"  Base:  {base_pos_pert}")
print(f"  Foot1: {foot1_pos_pert}")
print(f"  Foot2: {foot2_pos_pert}")

print(f"\nDeltas:")
print(f"  Base delta:  {base_pos_pert - base_pos_zero}")
print(f"  Foot1 delta: {foot1_pos_pert - foot1_pos_zero}")
print(f"  Foot2 delta: {foot2_pos_pert - foot2_pos_zero}")

jac_col = np.concatenate([
    (base_pos_pert - base_pos_zero) / epsilon,
    (foot1_pos_pert - foot1_pos_zero) / epsilon,
    (foot2_pos_pert - foot2_pos_zero) / epsilon
])

print(f"\nJacobian column for joint {j}:")
print(jac_col)

# Now test with fresh data object
print(f"\n" + "="*80)
print("REPEATING WITH FRESH DATA OBJECTS")
print("="*80)

data1 = mujoco.MjData(model)
data2 = mujoco.MjData(model)

# Set both to zero config
data1.qpos[:] = qpos_zero
data2.qpos[:] = qpos_zero

print(f"\nBefore forward kinematics:")
print(f"  data1.qpos[3] = {data1.qpos[3]}")
print(f"  data2.qpos[3] = {data2.qpos[3]}")

mujoco.mj_forward(model, data1)
base1 = data1.xpos[base_body_id].copy()
foot1_1 = data1.site_xpos[foot1_site_id].copy()

print(f"\nAfter FK on data1:")
print(f"  Foot1: {foot1_1}")

# Perturb data2
data2.qpos[3] += epsilon
mujoco.mj_forward(model, data2)
base2 = data2.xpos[base_body_id].copy()
foot1_2 = data2.site_xpos[foot1_site_id].copy()

print(f"\nAfter perturbing and FK on data2:")
print(f"  Foot1: {foot1_2}")

print(f"\nDelta: {foot1_2 - foot1_1}")
print(f"Jacobian column: {(foot1_2 - foot1_1) / epsilon}")

print("\n" + "="*80)
