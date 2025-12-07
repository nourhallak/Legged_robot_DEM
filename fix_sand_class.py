#!/usr/bin/env python3
"""
Add class="sand" to all sand particle geom elements
"""

xml_path = r"c:\Users\hplap\OneDrive\Desktop\Masters\1. Fall2025\MECH620 - Intermediate Dynamics\Project\DEM using Python\legged_robot_sand_shifted_low_friction.xml"

with open(xml_path, 'r') as f:
    content = f.read()

# Replace sand geom elements with class="sand"
# Pattern: <geom type="sphere" ... (for sand particles only)
import re

# Find all sand body definitions and add class="sand" to their geom
# Sand bodies have name="sand_*"

lines = content.split('\n')
new_lines = []

for line in lines:
    if '<body name="sand_' in line and '<geom type="sphere"' in line:
        # Add class="sand" to the geom element
        line = line.replace('<geom type="sphere"', '<geom type="sphere" class="sand"')
    new_lines.append(line)

new_content = '\n'.join(new_lines)

with open(xml_path, 'w') as f:
    f.write(new_content)

print("✓ Added class='sand' to all sand particle geom elements")
