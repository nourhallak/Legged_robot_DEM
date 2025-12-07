#!/usr/bin/env python3
"""
Reduce sand friction in the XML file from 0.8 to 0.4
"""

import re

xml_file = "legged_robot_sand_shifted.xml"

# Read the file
with open(xml_file, 'r') as f:
    content = f.read()

# Replace all sand particle friction values from 0.8 to 0.4
# Pattern: friction="0.8" in sand bodies
content = re.sub(r'(<body name="sand_.*?><inertial mass="0.05".*?<geom type="sphere".*?friction=")0\.8(")', r'\g<1>0.4\2', content)

# Write back
with open(xml_file, 'w') as f:
    f.write(content)

print("✓ Reduced all sand particle friction from 0.8 to 0.4")

# Verify
with open(xml_file, 'r') as f:
    verify = f.read()
    count_04 = len(re.findall(r'friction="0\.4"', verify))
    count_08 = len(re.findall(r'sand.*?friction="0\.8"', verify))
    print(f"Sand particles with friction=0.4: {count_04}")
    print(f"Sand particles with friction=0.8: {count_08}")
