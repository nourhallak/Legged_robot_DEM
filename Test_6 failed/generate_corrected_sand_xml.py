#!/usr/bin/env python3
"""Generate corrected XML with even higher sand and appropriate robot position."""
import numpy as np
import re

# Read original XML
with open('legged_robot_sand_shifted_low_friction.xml', 'r') as f:
    xml_content = f.read()

# FIRST: Remove all old sand particles (sand_*_*_* bodies)
sand_pattern = r'    <body name="sand_\d+_\d+_\d+"[^>]*>.*?</body>\n'
xml_content = re.sub(sand_pattern, '', xml_content, flags=re.DOTALL)

# Replace the floor - move it much higher to be at the bottom of sand  
old_floor = '<geom name="floor" type="plane" pos="0.316 0 0.431" size="2 2 0.1" rgba="0.8 0.9 0.8 1" friction="0.5" />'
new_floor = '<geom name="floor" type="plane" pos="0.316 0 0.48" size="2 2 0.1" rgba="0.8 0.9 0.8 1" friction="0.5" />'
xml_content = xml_content.replace(old_floor, new_floor)

# Replace robot hip position - keep at Z=0.400 but add floor support
# The feet should land on sand which sits at Z=0.48 (floor) to Z=0.51+ (surface)
old_hip_pos = '<body name="hip" pos="0.150 0.005 0.400">'
new_hip_pos = '<body name="hip" pos="0.150 0.005 0.475">'  # Hip MUCH lower for sand contact
xml_content = xml_content.replace(old_hip_pos, new_hip_pos)

# Generate sand particles - pack them densely from Z=0.48 to Z=0.51
# Floor at Z=0.48, sand particles on top
sand_particles = []
x_positions = np.arange(0.100, 0.400, 0.005)  # 5mm spacing
y_positions = np.array([-0.008, -0.003, 0.003, 0.008])  # 4 per column
# 3 layers: one sitting on floor, two on top
z_layers = [0.483, 0.492, 0.501]  # Top surface at ~0.504m

sand_id = 0
for z_idx, z_layer in enumerate(z_layers):
    for i, x in enumerate(x_positions):
        for j, y in enumerate(y_positions):
            name = f"sand_{z_idx}_{i}_{j}"
            # High friction (0.8) to keep sand particles from sliding
            particle = f'    <body name="{name}" pos="{x:.3f} {y:.3f} {z_layer:.3f}"><inertial mass="0.00001" pos="0 0 0" diaginertia="0.000000 0.000000 0.000000" /><geom type="sphere" size="0.003" rgba="0.76 0.70 0.55 1" friction="0.8 0.5 0.5" density="0.1" /></body>'
            sand_particles.append(particle)
            sand_id += 1

# Find the location to insert sand particles
insert_marker = '    <body name="hip"'
insert_pos = xml_content.find(insert_marker)

# Build new sand section
sand_section = '\n'.join(sand_particles) + '\n    '

# Insert sand particles before hip
xml_content = xml_content[:insert_pos] + sand_section + xml_content[insert_pos:]

# Write new XML
output_file = 'legged_robot_sand_fixed_surface.xml'
with open(output_file, 'w') as f:
    f.write(xml_content)

print(f"Generated {len(sand_particles)} sand particles")
print(f"Z layers: {z_layers}")
print(f"X range: {x_positions[0]:.3f} to {x_positions[-1]:.3f}")
print(f"Hip position: Z=0.475m (LOW - for strong sand contact)")
print(f"Expected foot contact: Deep into sand layers (strong pressure)")
print(f"Floor position: Z=0.48m (below sand)")
print(f"Sand friction: 0.8 (high, was 0.00001)")
print(f"\nWrote {output_file}")


