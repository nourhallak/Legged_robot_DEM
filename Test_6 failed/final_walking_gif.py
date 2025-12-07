#!/usr/bin/env python3
"""Save walking GIF with gravity keeping robot grounded."""
import mujoco
import mujoco.viewer
import numpy as np

print("="*70)
print("FINAL WALKING DEMO - Robot on Sand (Grounded)")
print("="*70)
print("Config: Hip Z=0.475m, Amplitude=0.1 (very weak), Gravity holds down")
print()

model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

frequency = 0.5
amplitude = 0.1  # VERY WEAK - let gravity do the work
num_actuators = len(data.ctrl)

print(f"Starting infinite walking viewer...")
print(f"Watch the robot walk continuously on sand")
print(f"Close window to stop")
print("="*70)
print()

with mujoco.viewer.launch_passive(model, data) as viewer:
    last_print = 0.0
    last_x = 0.0
    
    while viewer.is_running():
        t = data.time
        
        # Very weak control - gravity dominates
        phase = np.sin(2.0 * np.pi * frequency * t)
        phase_offset = np.sin(2.0 * np.pi * frequency * (t + 0.5/frequency))
        
        for i in range(num_actuators):
            if i < num_actuators // 2:
                data.ctrl[i] = -amplitude * max(phase, 0)
            else:
                data.ctrl[i] = -amplitude * max(phase_offset, 0)
        
        mujoco.mj_step(model, data)
        
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        
        viewer.sync()
        
        if t - last_print > 10.0:
            current_x = data.xpos[model.body('hip').id][0]
            current_z = data.xpos[model.body('hip').id][2]
            displacement = current_x - last_x
            velocity = displacement / 10.0
            
            print(f"T={t:6.1f}s | X={current_x:.4f}m | Z={current_z:.4f}m | V={velocity*1000:.1f}mm/s | Contacts={data.ncon}")
            
            last_print = t
            last_x = current_x

print("\nWalking complete!")

