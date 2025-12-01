#!/usr/bin/env python3
"""
Check the coordinate system and scaling of the robot model
"""
import numpy as np
import mujoco
import os

# Load the model
script_dir = os.path.dirname(os.path.abspath(__file__))
mjcf_path = os.path.join(script_dir, "legged_robot_ik.xml")

try:
    # Try modern API
    model = mujoco.MjModel.from_xml_path(mjcf_path)
except:
    # Fallback for older mujoco API
    model = mujoco.load_model_from_xml(open(mjcf_path).read())

data = mujoco.MjData(model)

print("=== MODEL INFORMATION ===")
print(f"Number of bodies: {model.nbody}")
print(f"Number of joints: {model.njnt}")
print(f"Number of sites: {model.nsite}")
print(f"Timestep: {model.opt.timestep}")

print("\n=== BODY HIERARCHY ===")
for i in range(model.nbody):
    body_name = model.body_names[i] if hasattr(model, 'body_names') else f"body_{i}"
    parent_id = model.body_parentid[i]
    parent_name = model.body_names[parent_id] if hasattr(model, 'body_names') else f"body_{parent_id}"
    
    print(f"Body {i}: {body_name}, Parent: {parent_name}")

print("\n=== JOINT INFORMATION ===")
for i in range(model.njnt):
    joint_name = model.jnt_names[i] if hasattr(model, 'jnt_names') else f"joint_{i}"
    joint_type = model.jnt_type[i]
    body_id = model.jnt_bodyid[i]
    body_name = model.body_names[body_id] if hasattr(model, 'body_names') else f"body_{body_id}"
    
    print(f"Joint {i}: {joint_name} (type={joint_type}) on body {body_name}")
    
    # Get joint limits
    addr = model.jnt_dofadr[i]
    if model.jnt_type[i] == 0:  # Free joint (6 DOF)
        print(f"  Free joint (6 DOF)")
    else:
        addr = model.jnt_dofadr[i]
        range_start = model.jnt_range[i, 0]
        range_end = model.jnt_range[i, 1]
        print(f"  Range: [{range_start:.4f}, {range_end:.4f}] rad = [{np.degrees(range_start):.1f}, {np.degrees(range_end):.1f}] deg")

print("\n=== SITE INFORMATION ===")
for i in range(model.nsite):
    site_name = model.site_names[i] if hasattr(model, 'site_names') else f"site_{i}"
    site_body = model.site_bodyid[i]
    body_name = model.body_names[site_body] if hasattr(model, 'body_names') else f"body_{site_body}"
    site_pos = model.site_pos[i]
    
    print(f"Site {i}: {site_name} on body {body_name} at local pos {site_pos}")

print("\n=== TEST DEFAULT POSE ===")
# Default pose - all zeros except floating base
mujoco.mj_kinematics(model, data)

for i in range(model.nsite):
    site_name = model.site_names[i] if hasattr(model, 'site_names') else f"site_{i}"
    site_pos_world = data.site_xpos[i] if hasattr(data, 'site_xpos') else np.array([0, 0, 0])
    print(f"Site {site_name} world position: {site_pos_world}")

print("\n=== GROUND LEVEL CHECK ===")
# Find geoms that might define ground
if hasattr(model, 'ngeom'):
    print(f"Number of geoms: {model.ngeom}")
    for i in range(model.ngeom):
        geom_type = model.geom_type[i]
        geom_size = model.geom_size[i]
        geom_pos = model.geom_pos[i]
        print(f"Geom {i}: type={geom_type}, size={geom_size}, pos={geom_pos}")
