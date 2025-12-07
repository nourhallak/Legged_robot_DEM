#!/usr/bin/env python3
"""
Inspect geometry names
"""

import mujoco

model = mujoco.MjModel.from_xml_path("legged_robot_sand.xml")

print(f"\nTotal geoms: {model.ngeom}\n")

# Show geom names for first and last few
print("First 20 geoms:")
for i in range(min(20, model.ngeom)):
    geom = model.geom(i)
    body_id = geom.bodyid
    body_name = model.body(body_id).name
    print(f"  {i}: {geom.name:20} (body: {body_name})")

print("\n...")
print(f"\nLast 20 geoms:")
for i in range(max(0, model.ngeom - 20), model.ngeom):
    geom = model.geom(i)
    body_id = geom.bodyid
    body_name = model.body(body_id).name
    print(f"  {i}: {geom.name:20} (body: {body_name})")

# Specific search for foot bodies and their geoms
print(f"\n\nFoot body geoms:")
for body_name in ['foot_1', 'foot_2']:
    try:
        body_id = model.body(body_name).id
        body_obj = model.body(body_id)
        print(f"\nBody: {body_name} (ID {body_id})")
        print(f"  geomnum: {body_obj.geomnum}")
        print(f"  geomadr: {body_obj.geomadr}")
        
        # Get geoms attached to this body
        for geom_id in range(body_obj.geomadr, body_obj.geomadr + body_obj.geomnum):
            geom = model.geom(geom_id)
            print(f"    Geom {geom_id}: {geom.name}")
    except Exception as e:
        print(f"  Error: {e}")
