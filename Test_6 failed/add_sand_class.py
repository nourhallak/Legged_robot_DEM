#!/usr/bin/env python3
"""Add class name to sand geoms for contact pair matching"""

import re

xml_file = "legged_robot_sand_shifted_low_friction.xml"

with open(xml_file, 'r') as f:
    content = f.read()

# Add class="sand" to all sand sphere geoms
content = re.sub(
    r'<geom type="sphere" size="0.0020" rgba="0.76 0.70 0.55 1" friction="0.02"',
    '<geom class="sand" type="sphere" size="0.0020" rgba="0.76 0.70 0.55 1" friction="0.02"',
    content
)

# Add class="foot" to foot meshes
content = re.sub(
    r'<geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="foot_1"',
    '<geom class="foot_1" type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="foot_1"',
    content
)

content = re.sub(
    r'<geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="foot_2"',
    '<geom class="foot_2" type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="foot_2"',
    content
)

with open(xml_file, 'w') as f:
    f.write(content)

print("[+] Added class names to sand and foot geometries")
print("[+] Sand geoms: class='sand'")
print("[+] Foot geoms: class='foot_1' and class='foot_2'")
