#!/usr/bin/env python3
"""
Check the robot model structure and site definitions
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from ik_simulation import load_model_with_assets
import mujoco

print("Loading model...")
model = load_model_with_assets()

print("\n" + "="*80)
print("MODEL STRUCTURE")
print("="*80)

print(f"\nTotal number of bodies: {model.nbody}")
print(f"Total number of joints: {model.njnt}")
print(f"Total number of dofs: {model.nv}")
print(f"Total number of sites: {model.nsite}")

print("\nBody names:")
for i in range(model.nbody):
    print(f"  {i}: {model.body(i).name}")

print("\nJoint info:")
for i in range(model.njnt):
    jnt = model.jnt_type[i]
    jnt_names = ["free", "ball", "slide", "hinge"]
    jnt_name = jnt_names[jnt] if jnt < len(jnt_names) else str(jnt)
    print(f"  Joint {i}: {model.joint(i).name:20} type={jnt_name:6} bodyid={model.jnt_bodyid[i]}")

print("\nSite info:")
for i in range(model.nsite):
    site = model.site(i)
    print(f"  Site {i}: {site.name:15} bodyid={model.site_bodyid[i]}")

print("\n" + "="*80)
print("CHECKING KINEMATIC CHAIN")
print("="*80)

data = mujoco.MjData(model)

# Test forward kinematics with identity pose
print("\nTest pose: All joints at 0")
data.qpos[:] = 0
mujoco.mj_forward(model, data)

foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id

print(f"\nBase position: {data.qpos[0:3]}")
print(f"Foot1 site position: {data.site_xpos[foot1_site_id]}")
print(f"Foot2 site position: {data.site_xpos[foot2_site_id]}")

# Check what happens when we move base
print("\n\nTest pose: Move base X by 0.01m")
data.qpos[:] = 0
data.qpos[0] = 0.01  # Move base X
mujoco.mj_forward(model, data)

print(f"Base position: {data.qpos[0:3]}")
print(f"Foot1 site position: {data.site_xpos[foot1_site_id]}")
print(f"Foot2 site position: {data.site_xpos[foot2_site_id]}")

# Check site bodyid - are feet attached to base or to links?
print(f"\n\nFoot1 site body id: {model.site_bodyid[foot1_site_id]} (name: {model.body(model.site_bodyid[foot1_site_id]).name})")
print(f"Foot2 site body id: {model.site_bodyid[foot2_site_id]} (name: {model.body(model.site_bodyid[foot2_site_id]).name})")
