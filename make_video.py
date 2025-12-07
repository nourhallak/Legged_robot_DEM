#!/usr/bin/env python3
"""
Generate video of robot walking on sand - using imageio
"""
import mujoco as mj
import numpy as np
import imageio

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

# Initial pose
data.qpos[model.joint("hip_link_2_1").id] = 0.0
data.qpos[model.joint("link_2_1_link_1_1").id] = -0.6
data.qpos[model.joint("link_1_1_foot_1").id] = 0.0
data.qpos[model.joint("hip_link_2_2").id] = 0.0
data.qpos[model.joint("link_2_2_link_1_2").id] = -0.6
data.qpos[model.joint("link_1_2_foot_2").id] = 0.0

mj.mj_forward(model, data)

print("Generating video of robot walking...")
print("Creating renderer...")

# Setup for rendering
renderer = mj.Renderer(model)
renderer.enable_depth_rendering()

# Prepare video frames list
frames = []
frame_count = 0

print("Simulating and rendering...")

sim_time = 0
t_end = 30.0

while sim_time < t_end:
    phase = (sim_time % 2.0) / 2.0
    
    # Leg 1 control
    if phase < 0.25:
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = 0.3
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
    elif phase < 0.5:
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = 0.0
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
    else:
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = -0.3
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.3
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
    
    # Leg 2 control (opposite phase)
    if phase < 0.25:
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = -0.3
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.3
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
    elif phase < 0.5:
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = 0.0
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
    else:
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = 0.3
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
    
    mj.mj_step(model, data)
    sim_time = data.time
    
    # Render every other frame to reduce video size (15fps instead of 30fps)
    if frame_count % 2 == 0:
        renderer.update_scene(data)
        img = renderer.render()
        frames.append(img)
    
    frame_count += 1
    
    if frame_count % 150 == 0:
        print(f"Processed {frame_count} simulation frames ({sim_time:.1f}s), recorded {len(frames)} video frames...")

print(f"\nWriting video file...")
imageio.mimsave('robot_walking.mp4', frames, fps=15)
print(f"Video saved: robot_walking.mp4")
print(f"Video frames: {len(frames)}")
print(f"Duration: {len(frames)/15:.1f}s")

renderer.close()
