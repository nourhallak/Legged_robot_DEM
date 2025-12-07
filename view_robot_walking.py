#!/usr/bin/env python3
"""Real-time visualization of robot walking on sand with MuJoCo viewer."""
import mujoco
import mujoco.viewer

# Load the corrected model
model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

# Parameters
frequency = 1.0  # Hz
amplitude = 0.3  # radians
num_actuators = len(data.ctrl)

# Create viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Simulation loop
    start_time = data.time
    sim_time = 0.0
    max_sim_time = 25.0  # Run for 25 seconds
    
    while viewer.is_running() and sim_time < max_sim_time:
        step_start = data.time
        
        # Apply walking gait
        t = data.time - start_time
        phase = mujoco.math.sin(2.0 * 3.14159 * frequency * t)
        
        for i in range(num_actuators):
            data.ctrl[i] = amplitude * phase
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Update viewer
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        
        viewer.sync()
        
        # Track time
        sim_time = data.time - start_time

print("Simulation complete! Viewer closed.")
print(f"Final time: {sim_time:.2f}s")
print(f"Robot position: X={data.xpos[model.body('hip').id][0]:.4f}m")
print(f"Robot height: Z={data.xpos[model.body('hip').id][2]:.4f}m")
