#!/usr/bin/env python3
"""Update sand friction to ultra-low for easy particle sliding"""

import re

xml_file = "legged_robot_sand_shifted_low_friction.xml"

with open(xml_file, 'r') as f:
    content = f.read()

# Change sand friction to very low (0.001) for sand-to-sand sliding
# Sand particles need to slide freely when foot pushes them
old_pattern = 'class="sand" type="sphere" size="0.0020" rgba="0.76 0.70 0.55 1" friction="0.02"'
new_pattern = 'class="sand" type="sphere" size="0.0020" rgba="0.76 0.70 0.55 1" friction="0.001 0.001 0.001"'

content = content.replace(old_pattern, new_pattern)

with open(xml_file, 'w') as f:
    f.write(content)

print("[+] Updated sand friction to 0.001 0.001 0.001")
print("[+] Sand particles will now slide easily when foot applies force")
