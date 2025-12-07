#!/usr/bin/env python3
"""Create slow-motion GIF of robot walking on sand."""
import mujoco
import numpy as np
from PIL import Image

print("Creating slow-motion GIF (0.5 Hz walking)...")

# Load model
model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

# Parameters - SLOW
frequency = 0.5  # Hz (half speed)
amplitude = 0.2  # radians (smaller movements)
num_actuators = len(data.ctrl)
sim_time_total = 30.0
dt = model.opt.timestep

# Rendering
renderer = mujoco.Renderer(model, height=480, width=640)

# Collect frames - save every 0.1s for smooth slow-mo playback
frames = []
save_interval = int(0.1 / dt)  # Every 0.1 seconds
total_steps = int(sim_time_total / dt)

print(f"Simulating {sim_time_total}s at slow speed...")
print(f"Capturing frame every {save_interval} steps (0.1s)...")

for step in range(total_steps):
    t = step * dt
    
    # Apply slow walking gait
    phase = np.sin(2.0 * np.pi * frequency * t)
    for i in range(num_actuators):
        data.ctrl[i] = amplitude * phase
    
    # Step simulation
    mujoco.mj_step(model, data)
    
    # Save frame
    if step % save_interval == 0:
        renderer.update_scene(data)
        frame_array = renderer.render()
        frame_pil = Image.fromarray(frame_array)
        frames.append(frame_pil)
        
        if len(frames) % 50 == 0:
            print(f"  Captured {len(frames)} frames ({t:.1f}s)...")

print(f"\nSaving GIF with {len(frames)} frames...")
frames[0].save(
    'robot_walking_slow_motion.gif',
    save_all=True,
    append_images=frames[1:],
    duration=150,  # 150ms per frame = 6.67 fps playback (very smooth)
    loop=0
)

print(f"✓ Saved: robot_walking_slow_motion.gif")
print(f"  Frames: {len(frames)}")
print(f"  Simulation duration: 30 seconds at 0.5 Hz")
print(f"  GIF playback: ~{len(frames)*150/1000:.1f}s at 6.67fps (smooth slow-mo)")
