#!/usr/bin/env python3
"""
Fix the XML defaults structure for MuJoCo compatibility
"""

xml_path = r"c:\Users\hplap\OneDrive\Desktop\Masters\1. Fall2025\MECH620 - Intermediate Dynamics\Project\DEM using Python\legged_robot_sand_shifted_low_friction.xml"

with open(xml_path, 'r') as f:
    content = f.read()

# Replace the problematic default structure
old_defaults = """  <default>
    <joint damping="0.001" armature="0.01" />
    <geom friction="0.5" density="1000" />
  </default>

  <default class="sand">
    <geom friction="0.00001 0.00001 0.00001" density="0.1" />
  </default>"""

new_defaults = """  <default>
    <joint damping="0.001" armature="0.01" />
    <geom friction="0.5" density="1000" />
  </default>"""

content = content.replace(old_defaults, new_defaults)

# Now update sand geoms to have explicit density and friction
# Replace class="sand" with explicit attributes
content = content.replace(
    '<geom type="sphere" class="sand" size="0.003" rgba="0.76 0.70 0.55 1" friction="0.00001 0.00001 0.00001" />',
    '<geom type="sphere" size="0.003" rgba="0.76 0.70 0.55 1" friction="0.00001 0.00001 0.00001" density="0.1" />'
)

# Also handle geoms that still have the old pattern without class
content = content.replace(
    '<geom type="sphere" size="0.003" rgba="0.76 0.70 0.55 1" friction="0.00001 0.00001 0.00001" />',
    '<geom type="sphere" size="0.003" rgba="0.76 0.70 0.55 1" friction="0.00001 0.00001 0.00001" density="0.1" />'
)

with open(xml_path, 'w') as f:
    f.write(content)

print("✓ Fixed XML default structure for MuJoCo compatibility")
print("  - Removed class='sand' from defaults")
print("  - Added explicit density='0.1' to all sand geoms")
