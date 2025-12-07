#!/usr/bin/env python3
"""
Test if robot is visible in viewer
"""

import mujoco
import mujoco.viewer

# Load model
model = mujoco.MjModel.from_xml_path("legged_robot_sand_shifted.xml")
data = mujoco.MjData(model)

print(f"Model loaded: {model.nbody} bodies, {model.ngeom} geoms")
print(f"Hip body index: {model.body('hip').id}")
print(f"Hip initial position: {data.xpos[model.body('hip').id]}")

# Try simple stiff pose
data.qpos[3] = 0.3  # left hip
data.qpos[4] = -0.5  # left knee
data.qpos[6] = -0.3  # right hip
data.qpos[7] = 0.5   # right knee

print("\nOpening viewer - robot should be visible in T-pose")
print("Press Ctrl+C or close window to exit\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(1000):
        mujoco.mj_step(model, data)
        viewer.sync()
        
        if i % 50 == 0:
            hip_pos = data.xpos[model.body('hip').id]
            print(f"Step {i}: Hip at {hip_pos}")

print("Done!")
