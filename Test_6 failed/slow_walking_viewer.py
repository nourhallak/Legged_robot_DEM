#!/usr/bin/env python3
"""Interactive MuJoCo viewer for robot walking on sand - SLOW MOTION."""
import mujoco
import mujoco.viewer
import numpy as np

# Load the corrected model
model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

print("Starting MuJoCo viewer with slow walking gait...")
print("Close the viewer window to exit.")
print()

# Parameters - SLOW DOWN
frequency = 0.5  # Hz (was 1.0) - slower walking
amplitude = 0.2  # radians (was 0.3) - smaller movements
num_actuators = len(data.ctrl)

# Create viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = data.time
    max_sim_time = 30.0  # Run for 30 seconds
    
    while viewer.is_running():
        step_start = data.time
        
        # Apply slow walking gait
        t = data.time - start_time
        phase = np.sin(2.0 * np.pi * frequency * t)
        
        for i in range(num_actuators):
            data.ctrl[i] = amplitude * phase
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Show contact points and forces
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        
        viewer.sync()
        
        # Stop after max time
        sim_time = data.time - start_time
        if sim_time > max_sim_time:
            break

print(f"\nSimulation complete!")
print(f"Final robot X position: {data.xpos[model.body('hip').id][0]:.4f}m")
print(f"Final robot Z position: {data.xpos[model.body('hip').id][2]:.4f}m")
