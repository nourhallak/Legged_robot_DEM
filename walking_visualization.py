#!/usr/bin/env python3
"""
Interactive visualization of robot walking ON TOP of sand
- Shows robot walking from start to end of sand bed
- Demonstrates sand contact and tight particle packing
- Robot stops at sand end
"""

import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path('legged_robot_sand_top_surface_v2.xml')
data = mujoco.MjData(model)

freq = 0.5
amplitude = 0.2
sand_x_end = 0.450
stop_threshold = 0.440

print("=" * 70)
print("INTERACTIVE VISUALIZATION")
print("=" * 70)
print("Robot walking on TOP of sand surface")
print("Close the window to exit")
print("=" * 70)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    
    last_print = 0
    
    while viewer.is_running() and data.time < 240:
        hip_id = model.body('hip').id
        x_pos = data.xpos[hip_id][0]
        
        # Control
        if x_pos > stop_threshold:
            data.ctrl[:] = np.zeros(model.nu)
        else:
            phase = np.sin(2.0 * np.pi * freq * data.time)
            phase_offset = np.sin(2.0 * np.pi * freq * (data.time + 0.5 / freq))
            
            for i in range(model.nu):
                if i < model.nu // 2:
                    data.ctrl[i] = -amplitude * max(phase, 0)
                else:
                    data.ctrl[i] = -amplitude * max(phase_offset, 0)
        
        mujoco.mj_step(model, data)
        
        # Print every 5 seconds
        if data.time - last_print > 5:
            z_pos = data.xpos[hip_id][2]
            vel = data.qvel[hip_id] if len(data.qvel) > hip_id else 0
            print(f"T={data.time:6.1f}s | X={x_pos:.4f}m | Z={z_pos:.4f}m | V={vel*1000:.1f}mm/s")
            last_print = data.time
        
        viewer.sync()

print("\nVisualization complete!")
