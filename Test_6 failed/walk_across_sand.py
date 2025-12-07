#!/usr/bin/env python3
"""Robot walks across sand from start to end."""
import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

print("="*70)
print("ROBOT WALKING ACROSS SAND BED")
print("="*70)
print("Sand layout:")
print("  Start: X=0.100m")
print("  End: X=0.400m")
print("  Robot starts at: X=0.150m")
print()
print("Gait:")
print("  Frequency: 0.3 Hz (slow walk)")
print("  Amplitude: 0.15 rad (weak - stays on sand)")
print("  Type: Trotting (alternating legs)")
print()
print("Behavior:")
print("  1. Walk forward across sand bed")
print("  2. Stop when reaching X=0.400m")
print("  3. Hold final position")
print()
print("Close window to exit")
print("="*70)
print()

frequency = 0.3  # Hz
amplitude = 0.15  # Weak control
num_actuators = len(data.ctrl)
sand_end = 0.400  # End of sand bed

with mujoco.viewer.launch_passive(model, data) as viewer:
    started = False
    reached_end = False
    last_print = 0.0
    
    while viewer.is_running():
        t = data.time
        current_x = data.xpos[model.body('hip').id][0]
        
        # Start walking when simulation begins
        if not started:
            started = True
            print(f"Starting position: X={current_x:.4f}m")
        
        # Check if reached end of sand
        if current_x >= sand_end and not reached_end:
            reached_end = True
            print(f"\n✓ Reached end of sand at X={current_x:.4f}m")
            print("Stopping motion...\n")
        
        # Apply controls only if not at end
        if not reached_end:
            phase = np.sin(2.0 * np.pi * frequency * t)
            phase_offset = np.sin(2.0 * np.pi * frequency * (t + 0.5/frequency))
            
            for i in range(num_actuators):
                if i < num_actuators // 2:
                    data.ctrl[i] = -amplitude * max(phase, 0)
                else:
                    data.ctrl[i] = -amplitude * max(phase_offset, 0)
        else:
            # Stop - zero controls
            for i in range(num_actuators):
                data.ctrl[i] = 0.0
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Visualization
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        
        viewer.sync()
        
        # Print progress
        if t - last_print > 2.0:
            if not reached_end:
                print(f"Time {t:6.1f}s | Position X={current_x:.4f}m | Contacts={data.ncon}")
            else:
                print(f"Time {t:6.1f}s | STOPPED at X={current_x:.4f}m | Holding position")
            last_print = t

print("\nSimulation complete - robot has walked to end of sand and stopped")
