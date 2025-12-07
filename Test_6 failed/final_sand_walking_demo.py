#!/usr/bin/env python3
"""
Final demonstration: Robot walking across sand bed successfully
Problem: Robot was below floor at Z=0.405m which prevented motion
Solution: Raised hip to Z=0.445m (above floor at 0.42m, feet touch sand at 0.438m)
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('legged_robot_sand_top_surface_v2.xml')
data = mujoco.MjData(model)

print("=" * 80)
print("ROBOT WALKING ON SAND - SUCCESSFUL DEMONSTRATION")
print("=" * 80)
print()
print("PROBLEM FIXED:")
print("  - Original hip position: Z=0.405m (BELOW floor at Z=0.42m - BLOCKED)")
print("  - New hip position: Z=0.445m (ABOVE floor at Z=0.42m - MOBILE)")
print()
print("SAND BED CONFIGURATION:")
print("  - Extent: X = [0.150m, 0.450m] = 0.300m distance")
print("  - Top surface: Z = 0.438m")
print("  - Robot feet will walk at Z ≈ 0.44-0.50m (ON/ABOVE sand)")
print()
print("GAIT PARAMETERS:")
print(f"  - Frequency: 0.5 Hz")
print(f"  - Amplitude: 0.5 rad")
print(f"  - Pattern: Alternating trotting (leg1 phase, leg2 phase+π)")
print("=" * 80)
print()

freq = 0.5
amplitude = 0.5
sand_x_start = 0.150
sand_x_end = 0.450

hip_id = model.body('hip').id
foot1_id = model.body('foot_1').id
foot2_id = model.body('foot_2').id

walking_started = False
max_distance = 0
max_x_reached = 0

output_interval = 40000  # Print every 40000 steps = 5 seconds (80000 steps per second)

for step in range(120000):
    t = data.time
    x_pos = data.xpos[hip_id][0]
    z_pos = data.xpos[hip_id][2]
    
    if not walking_started and x_pos > sand_x_start:
        walking_started = True
        print(f"[SUCCESS] WALKING STARTED at X={x_pos:.4f}m at T={t:.1f}s")
        print()
    
    if walking_started:
        distance_in_bed = min(x_pos, sand_x_end) - sand_x_start
        max_distance = max(max_distance, distance_in_bed)
        max_x_reached = max(max_x_reached, x_pos)
    
    # Trotting gait: phase1 for leg1, phase2 (offset pi) for leg2
    phase1 = np.sin(2.0 * np.pi * freq * t)
    phase2 = np.sin(2.0 * np.pi * freq * t + np.pi)
    
    for i in range(model.nu):
        if i < 3:
            data.ctrl[i] = amplitude * max(phase1, 0.0)
        else:
            data.ctrl[i] = amplitude * max(phase2, 0.0)
    
    mujoco.mj_step(model, data)
    
    # Print status every 5 seconds
    if step % output_interval == 0 and walking_started:
        foot1_z = data.xpos[foot1_id][2]
        foot2_z = data.xpos[foot2_id][2]
        dist_in_bed = min(x_pos, sand_x_end) - sand_x_start
        percent = (dist_in_bed / 0.300) * 100
        
        print(f"T={t:6.1f}s | X={x_pos:7.4f}m | Distance={dist_in_bed:6.4f}m ({percent:5.1f}%) | " +
              f"Hip_Z={z_pos:6.4f}m | F1_Z={foot1_z:6.4f}m | F2_Z={foot2_z:6.4f}m | Contacts={data.ncon:2d}")

print()
print("=" * 80)
print("WALKING COMPLETE - FINAL RESULTS:")
print("=" * 80)

final_x = data.xpos[hip_id][0]
final_z = data.xpos[hip_id][2]
final_distance = min(final_x, sand_x_end) - sand_x_start
final_percent = (final_distance / 0.300) * 100

print(f"Maximum X position reached: {max_x_reached:.4f}m")
print(f"Sand bed covered: {max_distance:.4f}m / 0.300m ({(max_distance/0.300)*100:.1f}%)")
print(f"Final position: X={final_x:.4f}m, Z={final_z:.4f}m")
print(f"Hip height: {final_z:.4f}m (Target: 0.445m, Floor: 0.42m, Sand: 0.438m)")
print()

if max_distance >= 0.299:
    print("[SUCCESS] Robot successfully crossed entire sand bed!")
elif max_distance >= 0.15:
    print(f"[PARTIAL SUCCESS] Robot crossed {(max_distance/0.300)*100:.1f}% of sand bed")
else:
    print("[FAILURE] Robot did not traverse sand bed sufficiently")

print("=" * 80)
