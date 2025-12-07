#!/usr/bin/env python3
"""
Debug: Check if base can move at all
"""

import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("legged_robot_sand.xml")
data = mujoco.MjData(model)

print("Initial state:")
print(f"  qpos[0] (base X): {data.qpos[0]:.6f}")
print(f"  xpos[0] (base world X): {data.xpos[0][0]:.6f}")
print(f"  qvel[0] (base X vel): {data.qvel[0]:.6f}")

# Apply forward force
data.xfrc_applied[0, 0] = 10.0  # 10N forward

# Step a few times
for step in range(1000):
    mujoco.mj_step(model, data)

print("\nAfter 1000 steps with 10N force:")
print(f"  qpos[0] (base X): {data.qpos[0]:.6f}")
print(f"  xpos[0] (base world X): {data.xpos[0][0]:.6f}")
print(f"  qvel[0] (base X vel): {data.qvel[0]:.6f}")
print(f"  time: {data.time:.6f}s")

if data.qpos[0] != 0.0:
    print("\n✓ Base CAN move in joint space (qpos changed)")
else:
    print("\n✗ Base cannot move - qpos[0] is fixed at 0")
