#!/usr/bin/env python3
"""
Fix robot height to be above floor and on sand.
The problem: Hip is below floor at Z=0.405m when floor is at Z=0.42m.
Solution: Raise hip to Z=0.445m so it's above floor but feet can still touch sand.
"""

import xml.etree.ElementTree as ET

# Parse the current XML
tree = ET.parse("legged_robot_sand_top_surface_v2.xml")
root = tree.getroot()

# Find and modify hip position
# Current: <body name="hip" pos="0.2 0.005 0.405">
# New: <body name="hip" pos="0.2 0.005 0.445"> (0.025m above floor at 0.42m)

for body in root.findall(".//body"):
    if body.get("name") == "hip":
        pos = body.get("pos").split()
        print(f"OLD hip position: {pos}")
        
        # Calculate new Z: 0.445m (0.025m clearance above floor at 0.42m)
        # This should allow feet (at ~0.435m when extended) to touch sand at 0.438m
        new_z = 0.445
        pos[2] = str(new_z)
        body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        
        print(f"NEW hip position: {body.get('pos')}")
        print(f"  Floor at Z=0.420m")
        print(f"  Hip at Z={new_z}m (clearance: {new_z - 0.420}m)")
        print(f"  Sand top at Z=0.438m")
        print(f"  Expected foot height when standing: ~0.433-0.437m (should touch sand)")

# Write out the new XML
output_path = "legged_robot_sand_top_surface_v2.xml"
tree.write(output_path)
print(f"\nWrote {output_path}")
