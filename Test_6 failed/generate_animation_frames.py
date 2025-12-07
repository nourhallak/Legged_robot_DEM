#!/usr/bin/env python3
"""Generate animation frames showing robot walking on corrected sand."""
import mujoco
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

print("Generating animation frames of robot walking on sand...")

# Load the corrected model
model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

# Create output directory
os.makedirs('animation_frames', exist_ok=True)

# Video parameters
dt_sim = model.opt.timestep  # 0.002
sim_time = 25.0
total_frames = int(sim_time / dt_sim)

# MuJoCo rendering
renderer = mujoco.Renderer(model, height=480, width=640)

# Simple walking gait
frequency = 1.0  # Hz
amplitude = 0.3  # radians
num_actuators = len(data.ctrl)

print(f"Simulating {sim_time}s = {total_frames} frames...")
print(f"Saving key frames to animation_frames/ directory...")

frame_count = 0
# Save every 50th frame (0.1s intervals) for viewing
save_interval = 50

for step in range(total_frames):
    t = step * dt_sim
    
    # Apply simple walking gait
    phase = np.sin(2 * np.pi * frequency * t)
    for i in range(num_actuators):
        data.ctrl[i] = amplitude * phase
    
    # Simulate
    mujoco.mj_step(model, data)
    
    # Save every Nth frame
    if step % save_interval == 0:
        # Render scene
        renderer.update_scene(data)
        frame = renderer.render()
        
        # Convert to PIL Image
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        # Add text
        text_info = [
            f"Time: {t:.2f}s",
            f"Hip Z: {data.xpos[model.body('hip').id][2]:.4f}m",
            "Robot Walking ON SAND Surface"
        ]
        
        for i, text in enumerate(text_info):
            draw.text((20, 20 + i*30), text, fill=(255, 255, 255))
        
        # Save frame
        filename = f"animation_frames/frame_{frame_count:04d}.png"
        img.save(filename)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  Saved {frame_count} frames ({t:.1f}s)...")

print(f"\n✓ Saved {frame_count} animation frames")
print(f"  Location: animation_frames/")
print(f"  Span: 0s to {(frame_count-1)*save_interval*dt_sim:.1f}s")
print(f"\nTo create video from frames, use:")
print(f"  ffmpeg -framerate 10 -i animation_frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p walking_animation.mp4")
