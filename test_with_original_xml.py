#!/usr/bin/env python3
"""
Test the trotting gait control with the original working XML to validate algorithm
"""

import mujoco
import numpy as np

# Load the original WORKING XML (that had the robot walking)
model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

freq = 0.5
amplitude = 0.2

print("Testing control algorithm with ORIGINAL WORKING XML")
print(f"Model has {model.nu} actuators")
print(f"Simulation will run for 30 seconds\n")

step_count = 0
last_print = 0

try:
    for step in range(15000):  # 30 seconds at 0.002 timestep
        t = data.time
        
        # Apply trotting control (same as in working version)
        phase = np.sin(2.0 * np.pi * freq * t)
        phase_offset = np.sin(2.0 * np.pi * freq * (t + 0.5 / freq))
        
        for i in range(model.nu):
            if i < model.nu // 2:
                data.ctrl[i] = -amplitude * max(phase, 0)
            else:
                data.ctrl[i] = -amplitude * max(phase_offset, 0)
        
        mujoco.mj_step(model, data)
        
        # Print every 5 seconds
        if t - last_print > 5:
            x_pos = data.xpos[model.body('hip').id][0]
            z_pos = data.xpos[model.body('hip').id][2]
            print(f"T={t:6.1f}s | X={x_pos:.4f}m | Z={z_pos:.4f}m | Contacts={data.ncon}")
            last_print = t
        
        step_count += 1

except KeyboardInterrupt:
    pass

print(f"\nCompleted {step_count} steps")
final_x = data.xpos[model.body('hip').id][0]
print(f"Final X position: {final_x:.4f}m")
