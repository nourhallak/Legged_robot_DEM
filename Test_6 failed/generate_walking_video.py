#!/usr/bin/env python3
"""Generate video of robot walking on corrected sand layer."""
import mujoco
import numpy as np
import imageio
import os

print("Generating video of robot walking on sand...")

# Load the corrected model
model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

# Video parameters
video_file = 'robot_walking_on_sand.mp4'
fps = 20
dt_sim = model.opt.timestep  # 0.002
sim_time = 25.0
total_frames = int(sim_time / dt_sim)

# MuJoCo rendering
renderer = mujoco.Renderer(model, height=720, width=960)

# Prepare frames list
frames = []

# Simple walking gait
frequency = 1.0  # Hz
amplitude = 0.3  # radians
num_actuators = len(data.ctrl)

print(f"Rendering {total_frames} simulation frames...")
print(f"Playback will be at {fps} fps = {total_frames/fps/25:.1f}x speed")

frame_count = 0
render_interval = int(1 / (fps * dt_sim))  # Only render frames to achieve desired fps

for step in range(total_frames):
    t = step * dt_sim
    
    # Apply simple walking gait
    phase = np.sin(2 * np.pi * frequency * t)
    for i in range(num_actuators):
        data.ctrl[i] = amplitude * phase
    
    # Simulate
    mujoco.mj_step(model, data)
    
    # Render every Nth frame for proper fps
    if step % render_interval == 0:
        # Render scene
        renderer.update_scene(data)
        frame = renderer.render()
        
        # Add text info to frame (numpy array manipulation)
        # Create a copy to mark up
        marked_frame = frame.copy()
        
        # Since we can't easily add text, just save the raw frame
        frames.append(frame)
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"  Rendered {frame_count} frames ({t:.1f}s of simulation)...")

# Write video using imageio
print(f"\nWriting video file (this may take a minute)...")
imageio.mimwrite(video_file, frames, fps=fps, codec='libx264')

print(f"\n✓ Video saved: {video_file}")
print(f"  Duration: {len(frames)/fps:.1f}s")
print(f"  Resolution: 960x720")
print(f"  FPS: {fps}")
print(f"  Frames: {len(frames)}")

