#!/usr/bin/env python3
"""Infinite loop walking viewer - watch robot push on sand continuously."""
import mujoco
import mujoco.viewer
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

print("="*70)
print("ROBOT WALKING ON SAND - INFINITE LOOP")
print("="*70)
print("The robot will walk infinitely in a repeating pattern.")
print()
print("Configuration:")
print("  Hip: Z=0.475m (pushing into sand)")
print("  Frequency: 0.2 Hz (5 seconds per complete cycle)")
print("  Amplitude: 0.5 rad (strong leg movements)")
print("  Sand contact: 1.3 contacts per frame")
print()
print("Watch for:")
print("  ✓ Legs pushing DOWN into sand")
print("  ✓ Sand particles deforming/moving")
print("  ✓ Forward motion from leg pushes")
print("  ✓ Smooth, continuous walking")
print()
print("Close the MuJoCo window to stop")
print("="*70)
print()

# Parameters - VERY SLOW, STRONG PUSH
frequency = 0.2  # Hz (5 seconds per cycle)
amplitude = 0.5  # Large amplitude for strong push
num_actuators = len(data.ctrl)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    cycle_count = 0
    cycle_time = 1.0 / frequency  # Time for one complete cycle
    
    while viewer.is_running():
        # Get current time
        t = data.time
        
        # Apply repeating sinusoidal gait (naturally infinite)
        phase = np.sin(2.0 * np.pi * frequency * t)
        
        # Apply controls to all actuators
        for i in range(num_actuators):
            data.ctrl[i] = amplitude * phase
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Show contact visualization
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        
        viewer.sync()
        
        # Track cycles (optional)
        new_cycle = int(t / cycle_time)
        if new_cycle > cycle_count:
            cycle_count = new_cycle
            hip_id = model.body('hip').id
            x_pos = data.xpos[hip_id][0]
            print(f"Cycle {cycle_count}: X={x_pos:.4f}m")

print("\nViewer closed - simulation stopped")
