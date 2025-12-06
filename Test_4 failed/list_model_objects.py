#!/usr/bin/env python3
"""
List all bodies in the model
"""

import mujoco

model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")

print("Bodies in model:")
for i in range(model.nbody):
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"  {i}: {body_name}")

print("\nSites in model:")
for i in range(model.nsite):
    site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    print(f"  {i}: {site_name}")

print("\nGeoms in model:")
for i in range(model.ngeom):
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    print(f"  {i}: {geom_name}")
