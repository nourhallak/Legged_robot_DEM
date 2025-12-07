#!/usr/bin/env python3
"""Fix sand inertial positions from global (0.200 0 0) to local (0 0 0) coordinates"""

import re

# Read the XML file
with open('legged_robot_sand_shifted.xml', 'r') as f:
    content = f.read()

# Count how many sand particles need fixing
sand_count = content.count('<body name="sand_')
print(f"Found {sand_count} sand particles")

# Replace all sand particle inertial positions: 0.200 0 0 -> 0 0 0
# Match: <inertial mass="0.05" pos="0.200 0 0"
original_count = content.count('pos="0.200 0 0" diaginertia="0.000000 0.000000 0.000000"')
print(f"Found {original_count} sand particles with wrong inertial pos")

# Replace the incorrect inertial positions
content = content.replace(
    '<inertial mass="0.05" pos="0.200 0 0" diaginertia="0.000000 0.000000 0.000000"/>',
    '<inertial mass="0.05" pos="0 0 0" diaginertia="0.000000 0.000000 0.000000"/>'
)

# Verify
new_count = content.count('<inertial mass="0.05" pos="0.200 0 0" diaginertia="0.000000 0.000000 0.000000"/>')
fixed_count = content.count('<inertial mass="0.05" pos="0 0 0" diaginertia="0.000000 0.000000 0.000000"/>')

print(f"After fix: {new_count} particles still broken, {fixed_count} particles fixed")

# Write back
with open('legged_robot_sand_shifted.xml', 'w') as f:
    f.write(content)

print("✓ Fixed legged_robot_sand_shifted.xml")
