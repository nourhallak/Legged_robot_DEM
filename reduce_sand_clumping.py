#!/usr/bin/env python3
"""Reduce sand clumping: increase particle size, reduce mass, add damping"""

import re

xml_file = "legged_robot_sand_shifted_low_friction.xml"

with open(xml_file, 'r') as f:
    content = f.read()

# 1. Increase sand particle size from 0.0020 to 0.0030 (more spacing)
content = re.sub(
    r'size="0\.0020"',
    'size="0.003"',
    content
)

# 2. Reduce sand mass from 0.0005 to 0.00025 (half again)
content = re.sub(
    r'mass="0\.0005"',
    'mass="0.00025"',
    content
)

# 3. Reduce damping globally for less sticking
# Find default section and update joint damping
content = re.sub(
    r'<joint damping="0\.0"',
    '<joint damping="0.001"',
    content
)

with open(xml_file, 'w') as f:
    f.write(content)

print("[+] Updated sand properties:")
print("    - Particle size: 0.0020 → 0.003 (more spacing)")
print("    - Particle mass: 0.0005 → 0.00025 (lighter)")
print("    - Joint damping: 0.0 → 0.001 (reduce sticking)")
