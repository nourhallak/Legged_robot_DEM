#!/usr/bin/env python3
"""Generate corrected XML with sand packed on TOP surface."""
import numpy as np
import re

# Read original XML
with open('legged_robot_sand_shifted_low_friction.xml', 'r') as f:
    xml_content = f.read()

# Remove all old sand particles
sand_pattern = r'    <body name="sand_\d+_\d+_\d+"[^>]*>.*?</body>\n'
xml_content = re.sub(sand_pattern, '', xml_content, flags=re.DOTALL)

# Set floor at Z=0.42m
old_floor = '<geom name="floor" type="plane" pos="0.316 0 0.431" size="2 2 0.1" rgba="0.8 0.9 0.8 1" friction="0.5" />'
new_floor = '<geom name="floor" type="plane" pos="0.316 0 0.42" size="2 2 0.1" rgba="0.8 0.9 0.8 1" friction="0.5" />'
xml_content = xml_content.replace(old_floor, new_floor)

# Set robot hip HIGH so feet land ON TOP of sand
old_hip_pos = '<body name="hip" pos="0.150 0.005 0.400">'
new_hip_pos = '<body name="hip" pos="0.150 0.005 0.52">'  # HIGH - feet on top
xml_content = xml_content.replace(old_hip_pos, new_hip_pos)

# Generate TIGHTLY PACKED sand particles ON TOP
# Sphere radius = 0.003m
# Contact distance = 2 * radius = 0.006m
# Pack them tightly: X spacing = 0.006m, Y spacing = 0.006m, Z spacing = 0.006m

sand_particles = []
radius = 0.003

# Sand goes from X=0.1 to X=0.4m
x_positions = np.arange(0.103, 0.397, 0.006)  # 0.006m spacing
y_positions = np.arange(-0.009, 0.010, 0.006)  # 0.006m spacing
# Stack particles: bottom layer on top of floor
z_base = 0.42 + radius  # Sits on floor
z_positions = [z_base, z_base + 0.006, z_base + 0.012]  # 3 layers tightly stacked

sand_id = 0
for z_idx, z in enumerate(z_positions):
    for i, x in enumerate(x_positions):
        for j, y in enumerate(y_positions):
            name = f"sand_{z_idx}_{i}_{j}"
            # HIGH friction to keep particles together
            particle = f'    <body name="{name}" pos="{x:.4f} {y:.4f} {z:.4f}"><inertial mass="0.00001" pos="0 0 0" diaginertia="0.000000 0.000000 0.000000" /><geom type="sphere" size="0.003" rgba="0.76 0.70 0.55 1" friction="0.9 0.5 0.5" density="0.1" /></body>'
            sand_particles.append(particle)
            sand_id += 1

# Insert sand particles before hip
insert_marker = '    <body name="hip"'
insert_pos = xml_content.find(insert_marker)
sand_section = '\n'.join(sand_particles) + '\n    '
xml_content = xml_content[:insert_pos] + sand_section + xml_content[insert_pos:]

# Write new XML
output_file = 'legged_robot_sand_top_surface.xml'
with open(output_file, 'w') as f:
    f.write(xml_content)

print(f"Generated {len(sand_particles)} sand particles")
print(f"Configuration:")
print(f"  Floor: Z=0.420m")
print(f"  Sand layers: Z=0.423m, Z=0.429m, Z=0.435m (on top surface)")
print(f"  Sand spacing: 0.006m (touching each other)")
print(f"  Robot hip: Z=0.520m (HIGH - feet walk ON TOP)")
print(f"  Sand friction: 0.9 (high - particles stick together)")
print(f"  X range: {x_positions[0]:.3f} to {x_positions[-1]:.3f}m")
print(f"  Y range: {y_positions[0]:.3f} to {y_positions[-1]:.3f}m")
print(f"\nWrote {output_file}")
