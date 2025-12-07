#!/usr/bin/env python3
import re

with open('legged_robot_sand_low_damping.xml', 'r') as f:
    content = f.read()

# Replace all sand positions - shift X by 0.2m forward  
def shift_sand_pos(match):
    x = float(match.group(1))
    y = match.group(2)
    z = match.group(3)
    x_new = x + 0.2
    return f'pos="{x_new:.3f} {y} {z}"'

new_content = re.sub(r'pos="([-\d.]+) ([-\d.]+) ([-\d.]+)"', shift_sand_pos, content)

with open('legged_robot_sand_shifted.xml', 'w') as f:
    f.write(new_content)

print('[+] Created legged_robot_sand_shifted.xml')
print('[+] Sand shifted +0.2m to positive X region')
print('[+] Robot at X=0 will now naturally walk FORWARD into sand')
