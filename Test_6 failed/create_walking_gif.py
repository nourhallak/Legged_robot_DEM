#!/usr/bin/env python3
"""Create an animated GIF of robot walking on sand."""
import mujoco
import numpy as np
from PIL import Image
import os

print("Creating animated GIF of robot walking on sand...")

# Load the corrected model
model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

# Parameters
frequency = 1.0  # Hz
amplitude = 0.3  # radians
num_actuators = len(data.ctrl)
sim_time_total = 25.0
dt = model.opt.timestep

# Rendering
renderer = mujoco.Renderer(model, height=480, width=640)

# Collect frames - save every 0.2s (100 steps) for smooth playback
frames = []
save_interval = int(0.2 / dt)  # Every 0.2 seconds
total_steps = int(sim_time_total / dt)

print(f"Simulating for {sim_time_total}s ({total_steps} steps)...")
print(f"Saving frame every {save_interval} steps (0.2s intervals)...")

for step in range(total_steps):
    t = step * dt
    
    # Apply walking gait
    phase = np.sin(2.0 * np.pi * frequency * t)
    for i in range(num_actuators):
        data.ctrl[i] = amplitude * phase
    
    # Step simulation
    mujoco.mj_step(model, data)
    
    # Save frame
    if step % save_interval == 0:
        renderer.update_scene(data)
        frame_array = renderer.render()
        
        # Convert to PIL Image
        frame_pil = Image.fromarray(frame_array)
        frames.append(frame_pil)
        
        if len(frames) % 50 == 0:
            print(f"  Captured {len(frames)} frames ({t:.1f}s)...")

print(f"\nSaving GIF with {len(frames)} frames...")
# Save as animated GIF
frames[0].save(
    'robot_walking_on_sand.gif',
    save_all=True,
    append_images=frames[1:],
    duration=200,  # 200ms per frame = 5 fps playback
    loop=0  # Loop forever
)

print(f"✓ Saved: robot_walking_on_sand.gif")
print(f"  Frames: {len(frames)}")
print(f"  Duration: {len(frames) * 0.2:.1f}s of simulation")
print(f"  Playback speed: {len(frames) * 200 / 1000:.1f}s at 5fps")

# Also save key frame as PNG for quick viewing
print(f"\nSaving key frame as PNG...")
frames[len(frames)//2].save('robot_walking_midway.png')
print(f"✓ Saved: robot_walking_midway.png (mid-simulation frame)")

print("\nTo view the GIF, open: robot_walking_on_sand.gif")
