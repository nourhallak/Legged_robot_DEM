#!/usr/bin/env python3
"""
Debug script to understand robot structure and forces.
"""

import mujoco
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path("legged_robot_sand.xml")
data = mujoco.MjData(model)

print("Robot Structure Analysis")
print("=" * 70)

# Print body information
print("\nBodies:")
for i in range(model.nbody):
    print(f"  Body {i}: {model.body(i).name}")

print("\nJoints:")
for i in range(model.njnt):
    joint = model.joint(i)
    print(f"  Joint {i}: {joint.name}")

print("\nActuators:")
for i in range(model.nu):
    actuator = model.actuator(i)
    print(f"  Actuator {i}: {actuator.name}")

print("\n" + "=" * 70)
print("Initial Configuration:")
print(f"  qpos shape: {data.qpos.shape}")
print(f"  qvel shape: {data.qvel.shape}")
print(f"  ctrl shape: {data.ctrl.shape}")
print(f"  xfrc_applied shape: {data.xfrc_applied.shape}")

print("\nInitial positions (qpos):")
for i in range(min(9, len(data.qpos))):
    print(f"  qpos[{i}]: {data.qpos[i]:.6f}")

print("\n" + "=" * 70)
print("Body ID for force application:")
print(f"  Body 1 (hip): {model.body(1).name}")
print(f"  This is where we apply external force")
print(f"  xfrc_applied[1] = {data.xfrc_applied[1]}")

# Check which joint controls the base X position
print("\n" + "=" * 70)
print("Base control joints:")
for i in range(min(3, model.njnt)):
    joint = model.joint(i)
    axis = model.jnt_axis[i]
    print(f"  Joint {i} ({joint.name}): axis={axis}, range=[{joint.range[0]:.2f}, {joint.range[1]:.2f}]")
