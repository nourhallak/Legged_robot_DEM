#!/usr/bin/env python3
"""
Modified XML generator - Remove base joint damping to allow forward motion
"""

def create_low_damping_xml():
    """Create XML with zero base damping for forward motion"""
    
    sand_xml = ""  # Will be replaced with actual sand
    
    full_xml = '''<?xml version="1.0" ?>
<mujoco model="Robot_Walking_on_Sand">
  <compiler angle="radian" coordinate="local"/>
  
  <option timestep="0.005" gravity="0 0 -9.81">
    <flag contact="enable"/>
  </option>
  
  <default>
    <joint damping="0.0" armature="0.01"/>
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
    
    <!-- Sand particles in 3 layers -->
    <!-- SAND_PARTICLES_PLACEHOLDER -->
    
    <!-- Robot -->
    <body name="hip" pos="0 0 0.44">
      <inertial mass="0.5" pos="0 0 0" diaginertia="0.001 0.001 0.001"/>
      <joint name="root_x" type="slide" axis="1 0 0" range="-0.5 2.0" damping="0.0"/>
      <joint name="root_y" type="slide" axis="0 1 0" range="-0.5 0.5" damping="0.0"/>
      <joint name="root_rz" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.0"/>
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
    # Get sand from existing XML
    import re
    
    with open("legged_robot_sand.xml", "r") as f:
        content = f.read()
    
    # Extract sand bodies
    sand_match = re.search(r'<!-- Sand particles.*?-->(.+?)<!-- Robot -->', content, re.DOTALL)
    if sand_match:
        sand_xml = sand_match.group(1).strip()
        
        # Create new XML with sand
        template = create_low_damping_xml()
        new_xml = template.replace('<!-- SAND_PARTICLES_PLACEHOLDER -->', sand_xml)
        
        with open("legged_robot_sand_low_damping.xml", "w") as f:
            f.write(new_xml)
        
        print("[+] Created legged_robot_sand_low_damping.xml (zero base damping for forward motion)")
    else:
        print("[-] Could not extract sand particles from existing XML")
