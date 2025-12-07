#!/usr/bin/env python3
"""Modify sand to be more fluid with lower density and better contact properties"""

xml_file = "legged_robot_sand_shifted_low_friction.xml"

with open(xml_file, 'r') as f:
    content = f.read()

# 1. Reduce sand density (less rigid packing)
content = content.replace('density="1000"', 'density="500"')

# 2. Make sand friction even lower (nearly frictionless)
content = content.replace('friction="0.001 0.001 0.001"', 'friction="0.0001 0.0001 0.0001"')

with open(xml_file, 'w') as f:
    f.write(content)

print("[+] Sand optimization for fluidity:")
print("    - Density: 1000 → 500 (less rigid)")
print("    - Friction: 0.001 → 0.0001 (nearly frictionless)")
print("[+] Sand particles should now flow more easily under foot pressure")
