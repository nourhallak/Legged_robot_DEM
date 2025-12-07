#!/usr/bin/env python3
"""
Final demonstration: Robot walks from beginning to end of sand, then stops.
- Sand bed: X=0.150 to 0.450m
- Robot walks ON TOP of tightly packed sand particles
- Stops when reaching end of sand bed
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('legged_robot_sand_top_surface_v2.xml')
data = mujoco.MjData(model)

# Gait parameters
freq = 0.5  # Hz
amplitude = 0.35  # rad (increased from 0.2 to enable forward motion)

# Sand bed boundaries
sand_x_start = 0.150
sand_x_end = 0.450
stop_threshold = 0.440  # Stop control when approaching end

print("=" * 70)
print("FINAL DEMONSTRATION: Robot Walking ON Sand Surface")
print("=" * 70)
print(f"Sand bed: X=[{sand_x_start}m, {sand_x_end}m]")
print(f"Sand surface: Z=0.426-0.438m (3 layers, tightly packed)")
print(f"Robot hip: Z=0.405m (walking on sand)")
print(f"Gait: Trotting at {freq} Hz, amplitude={amplitude} rad (stronger for contact)")
print("=" * 70)
print()

step_count = 0
last_print = 0
walking_started = False
reached_end = False

for step in range(120000):  # 240 seconds max
    t = data.time
    hip_id = model.body('hip').id
    x_pos = data.xpos[hip_id][0]
    z_pos = data.xpos[hip_id][2]
    
    # Mark when walking starts
    if not walking_started and x_pos > sand_x_start:
        walking_started = True
        print(f"[OK] WALKING STARTED at X={x_pos:.4f}m")
    
    # Apply control (stop at end)
    if x_pos > stop_threshold:
        # Stop control - robot will coast to a stop
        data.ctrl[:] = np.zeros(model.nu)
        if not reached_end:
            reached_end = True
            print(f"\n[OK] SAND END REACHED at X={x_pos:.4f}m")
    else:
        # Trotting gait control
        phase = np.sin(2.0 * np.pi * freq * t)
        phase_offset = np.sin(2.0 * np.pi * freq * (t + 0.5 / freq))
        
        for i in range(model.nu):
            if i < model.nu // 2:
                data.ctrl[i] = -amplitude * max(phase, 0)
            else:
                data.ctrl[i] = -amplitude * max(phase_offset, 0)
    
    mujoco.mj_step(model, data)
    
    # Print progress every 5 seconds
    if t - last_print > 5:
        vel = data.qvel[hip_id] if len(data.qvel) > hip_id else 0
        status = ""
        if not walking_started:
            status = "(initializing)"
        elif x_pos > stop_threshold:
            status = "(STOPPED)"
        else:
            status = f"(Distance: {x_pos-sand_x_start:.4f}m / {sand_x_end-sand_x_start:.4f}m)"
        
        print(f"T={t:6.1f}s | X={x_pos:.4f}m | Z={z_pos:.4f}m | V={vel*1000:.1f}mm/s | Contacts={data.ncon:2d} {status}")
        last_print = t
    
    step_count += 1
    
    # Exit if robot has stopped moving and reached the end
    if reached_end and t > last_print + 10:
        break

# Final statistics
print("\n" + "=" * 70)
print("FINAL STATISTICS")
print("=" * 70)
hip_id = model.body('hip').id
final_x = data.xpos[hip_id][0]
final_z = data.xpos[hip_id][2]
final_time = data.time

distance_walked = final_x - sand_x_start
if distance_walked > 0:
    duration = final_time
    avg_vel = distance_walked / duration if duration > 0 else 0
    print(f"Total time: {final_time:.1f} seconds")
    print(f"Distance walked ON SAND: {distance_walked:.4f}m")
    print(f"Average velocity: {avg_vel*1000:.2f} mm/s")
    print(f"Final position: X={final_x:.4f}m, Z={final_z:.4f}m")
    print(f"Final contacts: {data.ncon}")
    
    if reached_end:
        print(f"\n[SUCCESS] Robot walked from start to end of sand bed!")
        print(f"     Distance: {sand_x_end - sand_x_start:.3f}m")
        print(f"     Actual walk: {distance_walked:.3f}m")
    else:
        print(f"\n[INFO] Robot stopped before reaching sand end")
else:
    print("[ERROR] Robot did not walk forward")

print("=" * 70)
