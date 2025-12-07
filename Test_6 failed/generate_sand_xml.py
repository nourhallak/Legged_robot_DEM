#!/usr/bin/env python3
"""
Generate MuJoCo XML with 1000+ sand balls arranged in 3 layers.
Sand balls are spheres that touch each other and respond to gravity.
Particles are arranged in a tight grid so they touch and interact.
"""

import numpy as np

def generate_sand_xml(num_balls=1000, ball_radius=0.002):
    """
    Generate sand particle XML bodies.
    Arranges balls in 3 layers in a NARROW layout (4 particles wide, extended length).
    
    Spacing = 2 * radius (particles touch exactly)
    Layout: 4 particles in Y direction, extended in X (length) direction
    Layer heights: 0.445m, 0.460m, 0.475m (z)
    Ball radius: 0.002m (2mm) - reduced 2.5x from 5mm for finer sand resolution
    """
    
    spacing = 2 * ball_radius  # Particles touch exactly (0.004m)
    
    # Sand layers positioned so robot feet can rest on top
    # Floor is at Z=0.431m, sand must be ABOVE it  
    # Robot feet rest at Z=0.4504m (with hip at 0.44m and IK standing angles)
    # Place 3 layers with top at ~0.450m
    z_layers = [0.442, 0.446, 0.450]  # 3 layers, 4mm apart (particles touch vertically)
    
    # NARROW layout: 4 particles in Y, extended in X
    num_in_y = 4  # Fixed: 4 particles wide
    
    # Calculate balls in X direction to reach num_balls target
    balls_per_layer = num_balls // len(z_layers)
    num_in_x = balls_per_layer // num_in_y + (1 if balls_per_layer % num_in_y else 0)
    
    print(f"Generating sand particles...")
    print(f"  Target balls: {num_balls}")
    print(f"  Ball radius: {ball_radius*1000:.1f}mm")
    print(f"  Spacing (particles touching): {spacing*1000:.1f}mm")
    print(f"  Layout: {num_in_y} particles wide (Y) x {num_in_x} particles long (X)")
    print(f"  Balls per layer: ~{num_in_x * num_in_y}")
    
    # Center Y positions (4 particles centered around 0)
    y_min = -(num_in_y - 1) * spacing / 2
    y_max = (num_in_y - 1) * spacing / 2
    y_positions = np.linspace(y_min, y_max, num_in_y)
    
    # Extended X positions (for walking length) - START BEFORE ROBOT at X=-0.05
    x_min = -0.05
    x_max = x_min + (num_in_x - 1) * spacing
    x_positions = np.linspace(x_min, x_max, num_in_x)
    
    sand_bodies = []
    ball_count = 0
    
    for layer, z in enumerate(z_layers):
        layer_count = 0
        for i, x in enumerate(x_positions):
            for j, y in enumerate(y_positions):
                if ball_count >= num_balls:
                    break
                
                sand_bodies.append({
                    'name': f'sand_{layer}_{i}_{j}',
                    'x': x,
                    'y': y,
                    'z': z,
                    'radius': ball_radius
                })
                ball_count += 1
                layer_count += 1
            if ball_count >= num_balls:
                break
        
        print(f"  Layer {layer+1} (Z={z}m): {layer_count} particles")
    
    # Generate XML for all sand balls (with explicit inertial mass)
    xml_sand = ""
    for ball in sand_bodies:
        # Sand mass: 0.05 kg (50g)
        # Inertia for sphere: I = (2/5) * m * r^2
        radius = ball["radius"]
        mass = 0.05
        inertia = (2/5) * mass * (radius ** 2)
        
        xml_sand += f'    <body name="{ball["name"]}" pos="{ball["x"]:.3f} {ball["y"]:.3f} {ball["z"]:.3f}">'
        xml_sand += f'<inertial mass="{mass}" pos="0 0 0" diaginertia="{inertia:.6f} {inertia:.6f} {inertia:.6f}"/>'
        xml_sand += f'<geom type="sphere" size="{ball["radius"]:.4f}" rgba="0.76 0.70 0.55 1" friction="0.8"/>'
        xml_sand += f'</body>\n'
    
    print(f"[+] Generated {ball_count} sand particles total")
    print(f"[+] Dimensions: {x_positions[-1] - x_positions[0]:.2f}m long x {y_positions[-1] - y_positions[0]:.2f}m wide")
    print(f"[+] Sand X range: {x_positions[0]:.4f}m to {x_positions[-1]:.4f}m")
    print(f"[+] Sand Y range: {y_positions[0]:.4f}m to {y_positions[-1]:.4f}m")
    
    return xml_sand, ball_count

def create_full_xml(sand_xml, num_balls):
    """Create complete MuJoCo XML with robot and sand."""
    
    full_xml = f'''<?xml version="1.0" ?>
<mujoco model="Robot_Walking_on_Sand">
  <compiler angle="radian" coordinate="local"/>
  
  <option timestep="0.005" gravity="0 0 -9.81">
    <flag contact="enable"/>
  </option>
  
  <default>
    <joint damping="0.1" armature="0.01"/>
    <geom friction="1.0 0.1 0.1" density="1000"/>
  </default>

  <asset>
    <mesh name="hip" content_type="model/stl" file="Legged_robot/meshes/hip.STL"/>
    <mesh name="link_2_1" content_type="model/stl" file="Legged_robot/meshes/link_2_1.STL"/>
    <mesh name="link_1_1" content_type="model/stl" file="Legged_robot/meshes/link_1_1.STL"/>
    <mesh name="foot_1" content_type="model/stl" file="Legged_robot/meshes/foot_1.STL"/>
    <mesh name="link_2_2" content_type="model/stl" file="Legged_robot/meshes/link_2_2.STL"/>
    <mesh name="link_1_2" content_type="model/stl" file="Legged_robot/meshes/link_1_2.STL"/>
    <mesh name="foot_2" content_type="model/stl" file="Legged_robot/meshes/foot_2.STL"/>
  </asset>

  <worldbody>
    <!-- Ground -->
    <geom name="floor" type="plane" pos="0 0 0.431" size="2 2 0.1" rgba="0.8 0.9 0.8 1" friction="1.0"/>
    
    <!-- Sand particles in 3 layers ({num_balls} balls total) -->
{sand_xml}
    
    <!-- Robot -->
    <body name="hip" pos="0 0 0.44">
      <inertial mass="0.5" pos="0 0 0" diaginertia="0.001 0.001 0.001"/>
      <joint name="root_x" type="slide" axis="1 0 0" range="-0.5 2.0" damping="0.1"/>
      <joint name="root_y" type="slide" axis="0 1 0" range="-0.5 0.5" damping="0.1"/>
      <joint name="root_rz" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.1"/>
      <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="hip" conaffinity="1" contype="1"/>

      <body name="link_2_1" pos="-0.0056483 -0.0084654 0.034295" quat="0.657054 0.657056 -0.261301 0.261302">
        <inertial pos="0 -0.007 0.0015" quat="0.707107 0.707107 0 0" mass="0.07585399999999999" diaginertia="1.1249100000e-03 1.0972300000e-03 7.7529000000e-05"/>
        <joint name="hip_link_2_1" pos="0 0 0" axis="0 0 1" range="-1.57 1.57" actuatorfrcrange="-10 10"/>
        <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="link_2_1" conaffinity="1" contype="1"/>
        <body name="link_1_1" pos="0 -0.014 0" quat="0.980785 0 0 -0.195090">
          <inertial pos="0 -0.007 0.0015" quat="0.707107 0.707107 0 0" mass="0.087854" diaginertia="1.6961200000e-03 1.6548500000e-03 9.7940000000e-05"/>
          <joint name="link_2_1_link_1_1" pos="0 0 0" axis="0 0 1" range="-2.5 0.5" actuatorfrcrange="-10 10"/>
          <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="link_1_1" conaffinity="1" contype="1"/>
          <body name="foot_1" pos="0 -0.014 0" quat="1 0 0 0">
            <inertial pos="0.00377647 -0.00150811 0.0015" quat="0.508652 0.508652 0.491195 0.491195" mass="0.0529596" diaginertia="5.6740700000e-04 4.7329800000e-04 1.1471800000e-04"/>
            <joint name="link_1_1_foot_1" pos="0 0 0" axis="0 0 1" range="-1.57 1.57" actuatorfrcrange="-10 10"/>
            <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="foot_1" conaffinity="1" contype="1"/>
            <site name="foot1_site" pos="0 0 0" size="0.0001" rgba="1 0 0 0"/>
          </body>
        </body>
      </body>

      <body name="link_2_2" pos="-0.0056483 -0.0014654 0.034295" quat="0.657054 0.657056 -0.261301 0.261302">
        <inertial pos="0 -0.007 0.0015" quat="0.707107 0.707107 0 0" mass="0.07585399999999999" diaginertia="1.1249100000e-03 1.0972300000e-03 7.7529000000e-05"/>
        <joint name="hip_link_2_2" pos="0 0 0" axis="0 0 1" range="-1.57 1.57" actuatorfrcrange="-10 10"/>
        <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="link_2_2" conaffinity="1" contype="1"/>
        <body name="link_1_2" pos="0 -0.014 0" quat="0.980785 0 0 -0.195090">
          <inertial pos="0 -0.007 0.0015" quat="0.707107 0.707107 0 0" mass="0.087854" diaginertia="1.6961200000e-03 1.6548500000e-03 9.7940000000e-05"/>
          <joint name="link_2_2_link_1_2" pos="0 0 0" axis="0 0 1" range="-2.5 0.5" actuatorfrcrange="-10 10"/>
          <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="link_1_2" conaffinity="1" contype="1"/>
          <body name="foot_2" pos="0 -0.014 0" quat="1 0 0 0">
            <inertial pos="0.00377647 -0.00150811 0.0015" quat="0.508652 0.508652 0.491195 0.491195" mass="0.0529596" diaginertia="5.6740700000e-04 4.7329800000e-04 1.1471800000e-04"/>
            <joint name="link_1_2_foot_2" pos="0 0 0" axis="0 0 1" range="-1.57 1.57" actuatorfrcrange="-10 10"/>
            <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="foot_2" conaffinity="1" contype="1"/>
            <site name="foot2_site" pos="0 0 0" size="0.0001" rgba="0 0 1 0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="hip_link_2_1_motor" joint="hip_link_2_1" ctrllimited="true" ctrlrange="-1.0 1.0" gear="1"/>
    <motor name="link_2_1_link_1_1_motor" joint="link_2_1_link_1_1" ctrllimited="true" ctrlrange="-1.0 1.0" gear="1"/>
    <motor name="link_1_1_foot_1_motor" joint="link_1_1_foot_1" ctrllimited="true" ctrlrange="-1.0 1.0" gear="1"/>
    <motor name="hip_link_2_2_motor" joint="hip_link_2_2" ctrllimited="true" ctrlrange="-1.0 1.0" gear="1"/>
    <motor name="link_2_2_link_1_2_motor" joint="link_2_2_link_1_2" ctrllimited="true" ctrlrange="-1.0 1.0" gear="1"/>
    <motor name="link_1_2_foot_2_motor" joint="link_1_2_foot_2" ctrllimited="true" ctrlrange="-1.0 1.0" gear="1"/>
  </actuator>
</mujoco>
'''
    
    return full_xml

if __name__ == "__main__":
    sand_xml, num_sand = generate_sand_xml(num_balls=1000, ball_radius=0.002)
    full_xml = create_full_xml(sand_xml, num_sand)
    
    # Save to file
    with open("legged_robot_sand.xml", "w") as f:
        f.write(full_xml)
    
    print(f"[+] Saved legged_robot_sand.xml with {num_sand} sand balls")
