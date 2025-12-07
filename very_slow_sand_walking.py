#!/usr/bin/env python3
"""VERY SLOW interactive walking viewer - robot pushing on sand."""
import mujoco
import mujoco.viewer
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

print("="*70)
print("ROBOT WALKING ON SAND - SLOW MOTION VIEWER")
print("="*70)
print("Configuration:")
print("  Hip position: Z=0.475m (LOW)")
print("  Frequency: 0.2 Hz (VERY SLOW - 5 seconds per step)")
print("  Amplitude: 0.5 rad (STRONG leg movements)")
print("  Simulation time: 40 seconds")
print()
print("You should see:")
print("  ✓ Robot legs pushing DOWN into sand")
print("  ✓ Sand particles moving/compressing")
print("  ✓ Forward motion as feet push backward")
print("  ✓ Smooth, deliberate walking motion")
print()
print("Close the MuJoCo window to exit")
print("="*70)
print()

# Parameters - VERY SLOW, STRONG PUSH
frequency = 0.2  # Hz (very slow)
amplitude = 0.5  # Large amplitude
num_actuators = len(data.ctrl)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = data.time
    max_sim_time = 40.0  # 40 seconds
    
    while viewer.is_running():
        step_start = data.time
        
        # Apply very slow, strong walking gait
        t = data.time - start_time
        phase = np.sin(2.0 * np.pi * frequency * t)
        
        # Apply controls to all actuators
        for i in range(num_actuators):
            data.ctrl[i] = amplitude * phase
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Enable visualization
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        
        viewer.sync()
        
        # Stop after max time
        sim_time = data.time - start_time
        if sim_time > max_sim_time:
            break

# Print results
hip_id = model.body('hip').id
final_x = data.xpos[hip_id][0]
final_z = data.xpos[hip_id][2]

print("\n" + "="*70)
print("SIMULATION COMPLETE")
print("="*70)
print(f"Final robot X position: {final_x:.4f}m")
print(f"Final robot Z position: {final_z:.4f}m")
print(f"Forward displacement: ~{final_x - 0.150:.4f}m")
print("="*70)
