#!/usr/bin/env python3
"""
Fix the robot body structure - convert all global positions to local coordinates
The issue: Body positions are specified in global coords but should be LOCAL to parent
"""

import xml.etree.ElementTree as ET

# Parse the XML
tree = ET.parse('legged_robot_sand_shifted.xml')
root = tree.getroot()

# Find the hip body
worldbody = root.find('worldbody')
hip = worldbody.find("body[@name='hip']")
hip_pos = [0.150, 0, 0.44]

print(f"Hip position (global): {hip_pos}")

# Fix link_2_1 (left leg upper)
link_2_1 = hip.find("body[@name='link_2_1']")
if link_2_1 is not None:
    old_pos = link_2_1.get('pos')
    # Convert to local: subtract hip position from current global
    parts = [float(x) for x in old_pos.split()]
    local_pos = [parts[0] - hip_pos[0], parts[1] - hip_pos[1], parts[2] - hip_pos[2]]
    new_pos_str = f"{local_pos[0]:.6f} {local_pos[1]:.7f} {local_pos[2]:.6f}"
    link_2_1.set('pos', new_pos_str)
    print(f"link_2_1: {old_pos} → {new_pos_str}")

# Fix link_2_2 (right leg upper)
link_2_2 = hip.find("body[@name='link_2_2']")
if link_2_2 is not None:
    old_pos = link_2_2.get('pos')
    parts = [float(x) for x in old_pos.split()]
    local_pos = [parts[0] - hip_pos[0], parts[1] - hip_pos[1], parts[2] - hip_pos[2]]
    new_pos_str = f"{local_pos[0]:.6f} {local_pos[1]:.7f} {local_pos[2]:.6f}"
    link_2_2.set('pos', new_pos_str)
    print(f"link_2_2: {old_pos} → {new_pos_str}")

# Save back
tree.write('legged_robot_sand_shifted.xml', encoding='utf-8', xml_declaration=True)
print("\n✓ Fixed robot body positions to use local coordinates")
