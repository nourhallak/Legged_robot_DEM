#!/usr/bin/env python3
# Update sand properties for easier motion

with open('legged_robot_sand_shifted_low_friction.xml', 'r') as f:
    content = f.read()

# Reduce sand mass from 0.001 to 0.0005
content = content.replace('mass="0.001"', 'mass="0.0005"')

# Reduce sand friction from 0.05 to 0.02
content = content.replace('friction="0.05"', 'friction="0.02"')

with open('legged_robot_sand_shifted_low_friction.xml', 'w') as f:
    f.write(content)

print('[+] Updated sand mass: 0.001 -> 0.0005')
print('[+] Updated sand friction: 0.05 -> 0.02')
