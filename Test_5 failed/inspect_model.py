#!/usr/bin/env python3
"""
Debug script to inspect MuJoCo model structure
"""

import mujoco

model_path = "legged_robot_ik.xml"
model = mujoco.MjModel.from_xml_path(model_path)

print("\n" + "="*80)
print("MODEL STRUCTURE INSPECTION")
print("="*80)

print(f"\nJoints ({model.njnt}):")
for i in range(model.njnt):
    name = model.joint(i).name
    jtype = model.joint(i).type
    print(f"  {i}: {name:30s} (type: {jtype})")

print(f"\nSites ({model.nsite}):")
for i in range(model.nsite):
    name = model.site(i).name
    print(f"  {i}: {name}")

print(f"\nBodies ({model.nbody}):")
for i in range(model.nbody):
    name = model.body(i).name
    print(f"  {i}: {name}")

print(f"\nGeoms ({model.ngeom}):")
for i in range(model.ngeom):
    name = model.geom(i).name
    print(f"  {i}: {name}")

print("\n" + "="*80 + "\n")
