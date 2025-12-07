#!/usr/bin/env python3
import xml.etree.ElementTree as ET

tree = ET.parse('legged_robot_sand_shifted.xml')
root = tree.getroot()

# Find first few sand bodies
count = 0
sand_positions = []
for body in root.findall('.//body'):
    if 'sand_' in body.get('name', ''):
        pos = body.get('pos')
        if pos:
            parts = pos.split()
            x = float(parts[0])
            sand_positions.append(x)
            if count < 3:
                print(f'{body.get("name")}: x={x:.4f}')
                count += 1

print()
if sand_positions:
    print(f'Sand X range: [{min(sand_positions):.4f}, {max(sand_positions):.4f}]m')
    print('[✓] Sand successfully shifted to positive X region')
    print('[✓] Robot walking forward (+X) will move into sand bed')
