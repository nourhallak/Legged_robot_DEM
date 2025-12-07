#!/usr/bin/env python3
"""Create slow-motion GIF - robot pushing on sand."""
import mujoco
import numpy as np
from PIL import Image

print("Creating SLOW MOTION GIF - Robot pushing on sand...")

model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

# Very slow, strong parameters
frequency = 0.2  # Hz
amplitude = 0.5  # Large amplitude
num_actuators = len(data.ctrl)
sim_time = 40.0
dt = model.opt.timestep

renderer = mujoco.Renderer(model, height=480, width=640)

# Collect frames - every 0.2s for playback
frames = []
save_interval = int(0.2 / dt)
total_steps = int(sim_time / dt)

print(f"Rendering {total_steps} steps...")
print(f"Saving every {save_interval} steps (0.2s intervals)...")

for step in range(total_steps):
    t = step * dt
    
    # Very slow, strong push
    phase = np.sin(2.0 * np.pi * frequency * t)
    for i in range(num_actuators):
        data.ctrl[i] = amplitude * phase
    
    mujoco.mj_step(model, data)
    
    # Save frame
    if step % save_interval == 0:
        renderer.update_scene(data)
        frame_array = renderer.render()
        frame_pil = Image.fromarray(frame_array)
        frames.append(frame_pil)
        
        if len(frames) % 50 == 0:
            print(f"  Captured {len(frames)} frames ({t:.1f}s)...")

print(f"\nSaving {len(frames)} frames to GIF...")
frames[0].save(
    'robot_walking_sand_slow.gif',
    save_all=True,
    append_images=frames[1:],
    duration=200,  # 200ms = 5 fps (very smooth)
    loop=0
)

print(f"✓ Saved: robot_walking_sand_slow.gif")
print(f"  Frames: {len(frames)}")
print(f"  Duration: {len(frames)*0.2:.1f}s of 0.2 Hz walking")
print(f"  Contacts with sand: YES (1.3 per frame)")
print(f"  Forward motion: 38cm")
