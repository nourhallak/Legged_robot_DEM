#!/usr/bin/env python3
"""Trotting gait for actual walking on sand - push backward motion."""
import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

print("="*70)
print("ROBOT WALKING WITH TROTTING GAIT")
print("="*70)
print("Gait: Asymmetric push pattern")
print("  - Left legs push backward (negative)")
print("  - Right legs push backward (negative)")
print("  - Alternate timing for continuous motion")
print()
print("Configuration:")
print("  Hip: Z=0.475m (LOW - strong sand contact)")
print("  Frequency: 0.5 Hz (2 seconds per cycle)")
print("  Push amplitude: 0.7 rad")
print()
print("Expected: Continuous forward motion pushing on sand")
print("="*70)
print()

frequency = 0.5  # Hz
amplitude = 0.7
num_actuators = len(data.ctrl)

print(f"Number of actuators: {num_actuators}")
print(f"Actuator indices: 0-{num_actuators-1}")
print()

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    last_x = 0.0
    last_print_time = 0.0
    
    while viewer.is_running():
        t = data.time
        
        # Trotting gait: LEFT legs (indices 0,1) and RIGHT legs (indices 2,3) 
        # Push backward in alternating pattern
        phase = np.sin(2.0 * np.pi * frequency * t)
        phase_offset = np.sin(2.0 * np.pi * frequency * (t + 0.5/frequency))
        
        # Apply control
        for i in range(num_actuators):
            if i < num_actuators // 2:
                # Left legs - main push phase
                data.ctrl[i] = -amplitude * max(phase, 0)  # Only push, don't pull
            else:
                # Right legs - offset push phase
                data.ctrl[i] = -amplitude * max(phase_offset, 0)
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Show contact
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        
        viewer.sync()
        
        # Print progress
        if t - last_print_time > 2.0:
            current_x = data.xpos[model.body('hip').id][0]
            displacement = current_x - last_x
            velocity = displacement / 2.0
            print(f"Time {t:.1f}s: X={current_x:.4f}m, velocity={velocity*1000:.1f}mm/s, contacts={data.ncon}")
            last_x = current_x
            last_print_time = t

print("\nViewer closed")
