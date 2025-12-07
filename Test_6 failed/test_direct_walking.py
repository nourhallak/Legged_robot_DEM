#!/usr/bin/env python3
"""
Robot walking on sand with direct leg motor control
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('legged_robot_sand_top_surface_v2.xml')
data = mujoco.MjData(model)

print("=" * 70)
print("ROBOT WALKING ON SAND - DIRECT MOTOR CONTROL")
print("=" * 70)
print(f"Sand bed: X=[0.15m, 0.45m]")
print(f"Robot hip: Z=0.445m (CORRECTED - above floor at Z=0.42m)")
print(f"Actuators: {model.nu}")
print("=" * 70)
print()

freq = 0.5
amplitude = 0.5  # Higher amplitude for ground contact walking
sand_x_start = 0.150

step_count = 0
last_print = 0
walking_started = False

for step in range(120000):  # 240 seconds
    t = data.time
    hip_id = model.body('hip').id
    x_pos = data.xpos[hip_id][0]
    z_pos = data.xpos[hip_id][2]
    
    if not walking_started and x_pos > sand_x_start:
        walking_started = True
        print(f"[OK] WALKING STARTED at X={x_pos:.4f}m")
    
    # Simple trotting: push all motors backward with offset phases
    phase1 = np.sin(2.0 * np.pi * freq * t)
    phase2 = np.sin(2.0 * np.pi * freq * t + np.pi)
    
    # Leg 1: motors 0, 1, 2
    # Leg 2: motors 3, 4, 5
    for i in range(model.nu):
        if i < 3:
            # Leg 1: use phase1 (positive for forward)
            data.ctrl[i] = amplitude * max(phase1, 0.0)
        else:
            # Leg 2: use phase2 (offset by pi)
            data.ctrl[i] = amplitude * max(phase2, 0.0)
    
    mujoco.mj_step(model, data)
    
    # Print every 5 seconds
    if t - last_print > 5:
        vel = data.qvel[hip_id] if len(data.qvel) > hip_id else 0
        print(f"T={t:6.1f}s | X={x_pos:.4f}m | Z={z_pos:.4f}m | V={vel*1000:8.1f}mm/s | Contacts={data.ncon:2d}")
        last_print = t
    
    step_count += 1

# Final stats
print("\n" + "=" * 70)
hip_id = model.body('hip').id
final_x = data.xpos[hip_id][0]
distance = final_x - sand_x_start
print(f"Distance walked: {distance:.4f}m")
print(f"Contacts: {data.ncon}")
print("=" * 70)
