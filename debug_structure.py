#!/usr/bin/env python3
"""Check which body is which and where sites are attached"""
import mujoco
import numpy as np
import os
import re

def load_model_with_assets():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mjcf_path = os.path.join(script_dir, "legged_robot_ik.xml")
    
    with open(mjcf_path, 'r') as f:
        mjcf_content = f.read()
    
    meshes_dir = os.path.join(script_dir, "Legged_robot", "meshes")
    pattern = r'file="([^"]+\.STL)"'
    mesh_files = set(re.findall(pattern, mjcf_content))
    
    assets = {}
    for mesh_file in mesh_files:
        abs_path = os.path.join(meshes_dir, mesh_file)
        if os.path.exists(abs_path):
            with open(abs_path, 'rb') as f:
                assets[mesh_file] = f.read()
    
    model = mujoco.MjModel.from_xml_string(mjcf_content, assets=assets)
    return model

model = load_model_with_assets()
data = mujoco.MjData(model)

print("=== ROBOT STRUCTURE ===\n")

print("Bodies:")
for i in range(model.nbody):
    print(f"  {i}: {model.body(i).name}")

print("\nSites:")
for i in range(model.nsite):
    site_name = model.site(i).name
    body_id = model.site(i).bodyid
    body_name = model.body(body_id).name
    pos = model.site(i).pos
    print(f"  {i}: {site_name} -> body {body_id} ({body_name}) at pos {pos}")

print("\n=== TEST FORWARD KINEMATICS ===\n")

# Set up with base at 0.205
data.qpos[:3] = [0, 0, 0.205]
data.qpos[3:7] = [1, 0, 0, 0]
data.qpos[7:] = [0]*6

mujoco.mj_kinematics(model, data)

print("Config: hip at (0, 0, 0.205), all joints at zero")
print("\nBody positions (xpos):")
for i in range(model.nbody):
    print(f"  {model.body(i).name}: {data.xpos[i]}")

print("\nSite positions (xpos):")
for i in range(model.nsite):
    site_name = model.site(i).name
    print(f"  {site_name}: {data.site_xpos[i]}")
