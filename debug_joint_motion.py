#!/usr/bin/env python3
"""Debug what's actually happening with joint control."""

import mujoco
import numpy as np
import os

# Load current XML
xml_path = "legged_robot_sand_top_surface_v2.xml"
if not os.path.exists(xml_path):
    print(f"ERROR: {xml_path} not found")
    exit(1)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Get joint info
print("=== JOINT CONFIGURATION ===")
print(f"Total joints: {model.njnt}")
print(f"Total actuators: {model.nu}")
for i in range(min(6, model.njnt)):
    qpos_adr = model.jnt_qposadr[i]
    print(f"Joint {i}: qpos_adr={qpos_adr}")

print("\n=== INITIAL STATE ===")
print(f"Hip body Z: {data.body('hip').xpos[2]:.4f}")
print(f"Foot1 Z: {data.body('foot_1').xpos[2]:.4f}")
print(f"Foot2 Z: {data.body('foot_2').xpos[2]:.4f}")

print("\n=== ACTUATOR INFO ===")
print(f"Number of actuators: {model.nu}")
for i in range(min(6, model.nu)):
    print(f"Actuator {i}: ctrl_range = {model.actuator_ctrlrange[i]}")

print("\n=== TEST: Apply positive control to all joints ===")
mujoco.mj_resetData(model, data)
data.ctrl[:6] = 0.5  # Set all 6 actuators to 0.5

for step in range(100):
    mujoco.mj_step(model, data)
    if step % 20 == 0:
        print(f"Step {step}: Hip Z={data.body('hip').xpos[2]:.4f}, " +
              f"Foot1 Z={data.body('foot_1').xpos[2]:.4f}, " +
              f"Foot2 Z={data.body('foot_2').xpos[2]:.4f}, " +
              f"qpos[0:6]={data.qpos[0:6]}")

print("\n=== QPOS RANGE (joint angles) ===")
print(f"Initial qpos: {data.qpos[:6]}")
print(f"After 100 steps: {data.qpos[:6]}")

print("\n=== CHECKING CONTROL EFFECT ===")
mujoco.mj_resetData(model, data)
print("Initial qpos:", data.qpos[0:6])

# Apply maximum positive control
data.ctrl[:6] = 1.0
for step in range(50):
    mujoco.mj_step(model, data)

print("After ctrl=1.0 for 50 steps:", data.qpos[0:6])
print("Hip position:", data.body('hip').xpos)

# Try negative control
mujoco.mj_resetData(model, data)
data.ctrl[:6] = -1.0
for step in range(50):
    mujoco.mj_step(model, data)

print("After ctrl=-1.0 for 50 steps:", data.qpos[0:6])
print("Hip position:", data.body('hip').xpos)
