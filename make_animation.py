#!/usr/bin/env python3
"""
Generate animation of robot walking on sand - using PIL
"""
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

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

print("Generating animation of robot walking...")

# Setup for rendering
renderer = mj.Renderer(model)
renderer.enable_depth_rendering()

# Prepare frames list
frames = []
frame_count = 0

print("Simulating and rendering...")

sim_time = 0
t_end = 30.0
start_x = data.xpos[model.body("hip").id][0]

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
    
    # Render every other frame to reduce animation size (15fps from 30fps)
    if frame_count % 2 == 0:
        renderer.update_scene(data)
        img = renderer.render()
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img)
        
        # Add text overlay with time and position
        current_x = data.xpos[model.body("hip").id][0]
        displacement = current_x - start_x
        draw = ImageDraw.Draw(pil_img)
        text = f"Time: {sim_time:.1f}s | Displacement: {displacement:.4f}m"
        draw.text((10, 10), text, fill=(255, 255, 255))
        
        frames.append(pil_img)
    
    frame_count += 1
    
    if frame_count % 150 == 0:
        print(f"Processed {frame_count} simulation frames ({sim_time:.1f}s), recorded {len(frames)} animation frames...")

print(f"\nWriting animation file...")
frames[0].save(
    'robot_walking.gif',
    save_all=True,
    append_images=frames[1:],
    duration=67,  # 67ms per frame = ~15fps
    loop=0
)
print(f"Animation saved: robot_walking.gif")
print(f"Animation frames: {len(frames)}")
print(f"Duration: ~{len(frames)/15:.1f}s")
print(f"Format: GIF (animated)")

renderer.close()
