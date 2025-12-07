#!/usr/bin/env python3
"""Increase all actuator forces to 50N for stronger sand pushing"""

xml_file = "legged_robot_sand_shifted_low_friction.xml"

with open(xml_file, 'r') as f:
    content = f.read()

# Replace all actuatorfrcrange values from -20 20 to -50 50
content = content.replace('actuatorfrcrange="-20 20"', 'actuatorfrcrange="-50 50"')

with open(xml_file, 'w') as f:
    f.write(content)

print("[+] Updated all joint forces: -20 20 → -50 50")
print("[+] All joints now have 50N force capacity for stronger sand pushing")
