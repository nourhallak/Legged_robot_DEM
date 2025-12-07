#!/usr/bin/env python3
"""
Generate video of robot walking on sand
"""
import mujoco as mj
import numpy as np
import cv2
import mujoco.viewer as mjv

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

# Video writer
video_writer = None
frame_count = 0
fps = 30
width, height = 1280, 720

print(f"Video resolution: {width}x{height}@{fps}fps")

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
    
    # Render every frame (or every Nth frame if you want slower video)
    renderer.update_scene(data)
    
    # Get image from renderer
    img = renderer.render()
    
    # Initialize video writer on first frame
    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('robot_walking.mp4', fourcc, fps, (img.shape[1], img.shape[0]))
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Add timestamp text
    cv2.putText(img_bgr, f'Time: {sim_time:.2f}s', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img_bgr, f'Hip X: {data.body("hip").xpos[0]:.4f}m', (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    video_writer.write(img_bgr)
    frame_count += 1
    
    if frame_count % 150 == 0:
        print(f"Processed {frame_count} frames ({sim_time:.1f}s)...")

# Release video writer
if video_writer is not None:
    video_writer.release()
    print(f"\nVideo saved: robot_walking.mp4")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {frame_count/fps:.1f}s")

renderer.close()
