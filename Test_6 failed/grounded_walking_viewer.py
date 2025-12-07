#!/usr/bin/env python3
"""Walking on sand with WEAK control - stays grounded on sand."""
import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

print("="*70)
print("ROBOT WALKING ON SAND - GROUNDED VERSION")
print("="*70)
print("Control tuning:")
print("  Amplitude: 0.2 rad (WEAK - keeps robot on sand)")
print("  Frequency: 0.5 Hz (2 second cycle)")
print("  Gait: Trotting (alternating push)")
print()
print("Expected behavior:")
print("  ✓ Robot feet stay in CONTACT with sand")
print("  ✓ Strong push forces (2.5 contacts/frame)")
print("  ✓ Forward motion from leg pushes")
print("  ✓ Continuous walking without flying")
print()
print("Close window to exit")
print("="*70)
print()

frequency = 0.5
amplitude = 0.2  # WEAK control
num_actuators = len(data.ctrl)

with mujoco.viewer.launch_passive(model, data) as viewer:
    last_print = 0.0
    last_x = 0.0
    
    while viewer.is_running():
        t = data.time
        
        # Trotting gait with WEAK control
        phase = np.sin(2.0 * np.pi * frequency * t)
        phase_offset = np.sin(2.0 * np.pi * frequency * (t + 0.5/frequency))
        
        for i in range(num_actuators):
            if i < num_actuators // 2:
                data.ctrl[i] = -amplitude * max(phase, 0)
            else:
                data.ctrl[i] = -amplitude * max(phase_offset, 0)
        
        mujoco.mj_step(model, data)
        
        # Visualizations
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        
        viewer.sync()
        
        # Print progress
        if t - last_print > 5.0:
            current_x = data.xpos[model.body('hip').id][0]
            current_z = data.xpos[model.body('hip').id][2]
            displacement = current_x - last_x
            velocity = displacement / 5.0
            
            print(f"Time {t:6.1f}s | X={current_x:.4f}m | Z={current_z:.4f}m | V={velocity*1000:.1f}mm/s | Contacts={data.ncon}")
            
            last_print = t
            last_x = current_x

print("\nSimulation ended")
