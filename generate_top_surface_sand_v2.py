#!/usr/bin/env python3
"""
Generate improved top-surface sand configuration:
- Sand particles on TOP surface (Z=0.423-0.435m)
- Tight packing (0.006m spacing, particles touching)
- Robot starts at X=0.200m with initial contact
- Sand bed extends X=0.150 to 0.450m (larger range for walking)
- High friction for particle cohesion
"""

import numpy as np

# Configuration
floor_z = 0.420  # Rigid floor
sand_radius = 0.003  # Sphere radius
sand_spacing = 0.006  # Spacing = 2*radius (touching each other)
robot_hip_z = 0.52  # Robot hip height (ON TOP)
robot_hip_x_start = 0.200  # Robot starts here

# Sand bed extent
sand_x_min = 0.150  # Sand bed start (far from robot initial position)
sand_x_max = 0.450  # Sand bed end (plenty of room for walking)

# Y extent (narrow strip, robot walks along Y-axis mid-line)
sand_y_min = -0.009
sand_y_max = 0.009

# Z levels for 3 layers on top surface
z_layers = [
    floor_z + sand_spacing + 0.0,     # Layer 1: bottom, resting on floor
    floor_z + sand_spacing * 2,        # Layer 2: middle
    floor_z + sand_spacing * 3,        # Layer 3: top
]

print(f"Generating top-surface sand configuration:")
print(f"  Floor: Z={floor_z}m")
print(f"  Sand layers: {len(z_layers)} layers")
for i, z in enumerate(z_layers):
    print(f"    Layer {i+1}: Z={z:.4f}m")
print(f"  X range: {sand_x_min} to {sand_x_max}m")
print(f"  Y range: {sand_y_min} to {sand_y_max}m")
print(f"  Spacing: {sand_spacing}m (particles touching)")
print(f"  Robot hip starts at X={robot_hip_x_start}m, Z={robot_hip_z}m")

# Generate sand particles
xml_lines = [
    '<?xml version=\'1.0\' encoding=\'utf-8\'?>',
    '<mujoco model="Robot_Walking_on_Sand_TopSurface">',
    '  <compiler angle="radian" coordinate="local" />',
    '  ',
    '  <option timestep="0.002" gravity="0 0 -9.81">',
    '    <flag contact="enable" />',
    '  </option>',
    '  ',
    '  <default>',
    '    <joint damping="0.001" armature="0.01" />',
    '    <geom friction="0.5" density="1000" />',
    '  </default>',
    '',
    '  <asset>',
    '    <mesh name="hip" content_type="model/stl" file="Legged_robot/meshes/hip.STL" />',
    '    <mesh name="link_2_1" content_type="model/stl" file="Legged_robot/meshes/link_2_1.STL" />',
    '    <mesh name="link_1_1" content_type="model/stl" file="Legged_robot/meshes/link_1_1.STL" />',
    '    <mesh name="foot_1" content_type="model/stl" file="Legged_robot/meshes/foot_1.STL" />',
    '    <mesh name="link_2_2" content_type="model/stl" file="Legged_robot/meshes/link_2_2.STL" />',
    '    <mesh name="link_1_2" content_type="model/stl" file="Legged_robot/meshes/link_1_2.STL" />',
    '    <mesh name="foot_2" content_type="model/stl" file="Legged_robot/meshes/foot_2.STL" />',
    '  </asset>',
    '',
    '  <worldbody>',
    '    ',
    f'    <geom name="floor" type="plane" pos="0.300 0 {floor_z}" size="2 2 0.1" rgba="0.8 0.9 0.8 1" friction="0.5" />',
    '    ',
]

# Generate sand bodies
particle_count = 0
x_positions = np.arange(sand_x_min, sand_x_max, sand_spacing)
y_positions = np.arange(sand_y_min, sand_y_max + sand_spacing, sand_spacing)

for layer_idx, z in enumerate(z_layers):
    for x_idx, x in enumerate(x_positions):
        for y_idx, y in enumerate(y_positions):
            name = f"sand_{layer_idx}_{x_idx}_{y_idx}"
            xml_lines.append(
                f'    <body name="{name}" pos="{x:.4f} {y:.4f} {z:.4f}">'
                f'<inertial mass="0.00001" pos="0 0 0" diaginertia="0.000000 0.000000 0.000000" />'
                f'<geom type="sphere" size="{sand_radius:.3f}" rgba="0.76 0.70 0.55 1" friction="0.9 0.5 0.5" density="0.1" />'
                f'</body>'
            )
            particle_count += 1

print(f"Generated {particle_count} sand particles")

# Add robot
xml_lines.extend([
    f'    <body name="hip" pos="{robot_hip_x_start} 0.005 {robot_hip_z}">',
    '      <inertial mass="0.5" pos="0 0 0" diaginertia="0.001 0.001 0.001" />',
    '      <joint name="root_x" type="slide" axis="1 0 0" range="0.1 0.80" damping="0.0" />',
    '      <joint name="root_y" type="slide" axis="0 1 0" range="-0.01 0.01" damping="10.0" />',
    '      <joint name="root_rz" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.0" />',
    '      <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="hip" conaffinity="1" contype="1" />',
    '',
    '      <body name="link_2_1" pos="-0.0056483 -0.0084654 0.034295" quat="0.657054 0.657056 -0.261301 0.261302">',
    '        <inertial pos="0 -0.007 0.0015" quat="0.707107 0.707107 0 0" mass="0.07585399999999999" diaginertia="1.1249100000e-03 1.0972300000e-03 7.7529000000e-05" />',
    '        <joint name="hip_link_2_1" pos="0 0 0" axis="0 0 1" range="-1.57 1.57" actuatorfrcrange="-200 200" />',
    '        <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="link_2_1" conaffinity="1" contype="1" />',
    '        <body name="link_1_1" pos="0 -0.014 0" quat="0.980785 0 0 -0.195090">',
    '          <inertial pos="0 -0.007 0.0015" quat="0.707107 0.707107 0 0" mass="0.087854" diaginertia="1.6961200000e-03 1.6548500000e-03 9.7940000000e-05" />',
    '          <joint name="link_2_1_link_1_1" pos="0 0 0" axis="0 0 1" range="-2.5 0.5" actuatorfrcrange="-200 200" />',
    '          <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="link_1_1" conaffinity="1" contype="1" />',
    '          <body name="foot_1" pos="0 -0.014 0" quat="1 0 0 0">',
    '            <inertial pos="0.00377647 -0.00150811 0.0015" quat="0.508652 0.508652 0.491195 0.491195" mass="0.0529596" diaginertia="5.6740700000e-04 4.7329800000e-04 1.1471800000e-04" />',
    '            <joint name="link_1_1_foot_1" pos="0 0 0" axis="0 0 1" range="-1.57 1.57" actuatorfrcrange="-200 200" />',
    '            <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="foot_1" conaffinity="1" contype="1" friction="0.8 0.1 0.1" />',
    '            <site name="foot1_site" pos="0 0 0" size="0.0001" rgba="1 0 0 0" />',
    '          </body>',
    '        </body>',
    '      </body>',
    '',
    '      <body name="link_2_2" pos="-0.0056483 -0.0014654 0.034295" quat="0.657054 0.657056 -0.261301 0.261302">',
    '        <inertial pos="0 -0.007 0.0015" quat="0.707107 0.707107 0 0" mass="0.07585399999999999" diaginertia="1.1249100000e-03 1.0972300000e-03 7.7529000000e-05" />',
    '        <joint name="hip_link_2_2" pos="0 0 0" axis="0 0 1" range="-1.57 1.57" actuatorfrcrange="-200 200" />',
    '        <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="link_2_2" conaffinity="1" contype="1" />',
    '        <body name="link_1_2" pos="0 -0.014 0" quat="0.980785 0 0 -0.195090">',
    '          <inertial pos="0 -0.007 0.0015" quat="0.707107 0.707107 0 0" mass="0.087854" diaginertia="1.6961200000e-03 1.6548500000e-03 9.7940000000e-05" />',
    '          <joint name="link_2_2_link_1_2" pos="0 0 0" axis="0 0 1" range="-2.5 0.5" actuatorfrcrange="-200 200" />',
    '          <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="link_1_2" conaffinity="1" contype="1" />',
    '          <body name="foot_2" pos="0 -0.014 0" quat="1 0 0 0">',
    '            <inertial pos="0.00377647 -0.00150811 0.0015" quat="0.508652 0.508652 0.491195 0.491195" mass="0.0529596" diaginertia="5.6740700000e-04 4.7329800000e-04 1.1471800000e-04" />',
    '            <joint name="link_1_2_foot_2" pos="0 0 0" axis="0 0 1" range="-1.57 1.57" actuatorfrcrange="-200 200" />',
    '            <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="foot_2" conaffinity="1" contype="1" friction="0.8 0.1 0.1" />',
    '            <site name="foot2_site" pos="0 0 0" size="0.0001" rgba="0 0 1 0" />',
    '          </body>',
    '        </body>',
    '      </body>',
    '    </body>',
    '  </worldbody>',
    '',
    '  <actuator>',
    '    <motor name="hip_link_2_1_motor" joint="hip_link_2_1" ctrllimited="true" ctrlrange="-1.0 1.0" gear="1" />',
    '    <motor name="link_2_1_link_1_1_motor" joint="link_2_1_link_1_1" ctrllimited="true" ctrlrange="-1.0 1.0" gear="1" />',
    '    <motor name="link_1_1_foot_1_motor" joint="link_1_1_foot_1" ctrllimited="true" ctrlrange="-1.0 1.0" gear="1" />',
    '    <motor name="hip_link_2_2_motor" joint="hip_link_2_2" ctrllimited="true" ctrlrange="-1.0 1.0" gear="1" />',
    '    <motor name="link_2_2_link_1_2_motor" joint="link_2_2_link_1_2" ctrllimited="true" ctrlrange="-1.0 1.0" gear="1" />',
    '    <motor name="link_1_2_foot_2_motor" joint="link_1_2_foot_2" ctrllimited="true" ctrlrange="-1.0 1.0" gear="1" />',
    '  </actuator>',
    '</mujoco>',
])

xml_content = '\n'.join(xml_lines)

with open('legged_robot_sand_top_surface_v2.xml', 'w') as f:
    f.write(xml_content)

print(f"\nWrote legged_robot_sand_top_surface_v2.xml with {particle_count} sand particles")
