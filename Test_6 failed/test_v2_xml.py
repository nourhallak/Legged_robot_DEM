#!/usr/bin/env python3
"""
Test walking on improved top-surface sand (v2)
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('legged_robot_sand_top_surface_v2.xml')
data = mujoco.MjData(model)

freq = 0.5
amplitude = 0.2

print("Testing IMPROVED top-surface sand configuration (v2)")
print(f"Model has {model.nu} actuators")
print(f"Running for 60 seconds...\n")

step_count = 0
last_print = 0

try:
    for step in range(30000):  # 60 seconds
        t = data.time
        
        # Apply trotting control
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
            hip_id = model.body('hip').id
            x_pos = data.xpos[hip_id][0]
            z_pos = data.xpos[hip_id][2]
            vel = data.qvel[hip_id] if len(data.qvel) > hip_id else 0
            print(f"T={t:6.1f}s | X={x_pos:.4f}m | Z={z_pos:.4f}m | V={vel*1000:.1f}mm/s | Contacts={data.ncon:2d}")
            last_print = t
        
        step_count += 1
        
        if step % 5000 == 0 and step > 0:
            print(f"  ... {step} steps completed ...")

except KeyboardInterrupt:
    pass

hip_id = model.body('hip').id
final_x = data.xpos[hip_id][0]
final_z = data.xpos[hip_id][2]
print(f"\nCompleted {step_count} steps")
print(f"Final X: {final_x:.4f}m, Z: {final_z:.4f}m")
