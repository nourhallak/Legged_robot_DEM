#!/usr/bin/env python3
"""Make sand ultra-light and frictionless to enable free flow"""

xml_file = "legged_robot_sand_shifted_low_friction.xml"

with open(xml_file, 'r') as f:
    content = f.read()

# 1. Reduce sand mass to almost nothing (0.00001 kg = 10 grams total per particle)
content = content.replace('mass="0.00025"', 'mass="0.00001"')

# 2. Keep friction ultra low
content = content.replace('friction="0.0001 0.0001 0.0001"', 'friction="0.00001 0.00001 0.00001"')

# 3. Reduce density even more
content = content.replace('density="500"', 'density="100"')

with open(xml_file, 'w') as f:
    f.write(content)

print("[+] Sand ultra-light mode:")
print("    - Particle mass: 0.00025 → 0.00001 kg (ultra-light)")
print("    - Friction: 0.0001 → 0.00001 (virtually frictionless)")
print("    - Density: 500 → 100")
print("[+] Sand particles will now move with minimal resistance")
