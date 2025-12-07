#!/usr/bin/env python3
"""Remove class attributes and fix XML"""

xml_file = "legged_robot_sand_shifted_low_friction.xml"

with open(xml_file, 'r') as f:
    content = f.read()

# Remove class attributes that cause errors
content = content.replace('class="sand" ', '')
content = content.replace('class="foot_1" ', '')
content = content.replace('class="foot_2" ', '')

with open(xml_file, 'w') as f:
    f.write(content)

print("[+] Removed invalid class attributes")
