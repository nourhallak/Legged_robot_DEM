#!/usr/bin/env python3
"""
Test maximum forward speed without leg control
"""

import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("legged_robot_sand.xml")

# Load IK standing pose
ik_left_hip = np.load("ik_left_hip.npy")
ik_left_knee = np.load("ik_left_knee.npy")
ik_left_ankle = np.load("ik_left_ankle.npy")

print("Testing forward push without leg control:\n")

# Test various force levels
for test_force in [1.0, 5.0, 10.0, 20.0, 50.0]:
    data = mujoco.MjData(model)
    
    # Set standing pose
    data.qpos[0:3] = [0, 0, 0.44]  # Base
    data.qpos[3] = ik_left_hip[0]  # Left hip
    data.qpos[4] = ik_left_knee[0]
    data.qpos[5] = ik_left_ankle[0]
    data.qpos[6] = ik_left_hip[0]  # Right hip (mirror)
    data.qpos[7] = ik_left_knee[0]
    data.qpos[8] = ik_left_ankle[0]
    
    mujoco.mj_forward(model, data)
    initial_x = data.xpos[model.body('hip').id][0]
    
    for step in range(2000):  # 10 seconds
        data.xfrc_applied[0, 0] = test_force
        mujoco.mj_step(model, data)
    
    final_x = data.xpos[model.body('hip').id][0]
    distance = final_x - initial_x
    time = data.time
    avg_vel = distance / time
    
    print(f"Force: {test_force:5.1f}N -> Distance: {distance:+.4f}m, Avg vel: {avg_vel:+.6f} m/s")
