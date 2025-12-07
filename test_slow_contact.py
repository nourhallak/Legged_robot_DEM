#!/usr/bin/env python3
"""Test walking with VERY slow gait that actually pushes on sand."""
import mujoco
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

# VERY SLOW parameters
frequency = 0.2  # Hz (very slow - 1 step every 5 seconds)
amplitude = 0.5  # Large amplitude to push hard
num_actuators = len(data.ctrl)
sim_time = 30.0
dt = model.opt.timestep

print("Testing VERY SLOW walking with strong pushing...")
print(f"Frequency: {frequency} Hz (very slow)")
print(f"Amplitude: {amplitude} rad (large movements)")
print()

times = []
positions = []
contacts = []
foot_z = []

for step in range(int(sim_time / dt)):
    t = step * dt
    
    # Very slow, strong push gait
    phase = np.sin(2.0 * np.pi * frequency * t)
    
    for i in range(num_actuators):
        data.ctrl[i] = amplitude * phase
    
    mujoco.mj_step(model, data)
    
    # Record every 10 steps (0.02s)
    if step % 10 == 0:
        times.append(t)
        positions.append(data.xpos[model.body('hip').id][0])
        contacts.append(data.ncon)
        foot_z.append(data.xpos[model.body('foot_1').id][2])

print("="*60)
print("CONTACT AND MOTION ANALYSIS")
print("="*60)
print(f"Total displacement: {positions[-1] - positions[0]:.4f}m")
print(f"Average contacts per frame: {np.mean(contacts):.1f}")
print(f"Max contacts: {np.max(contacts)}")
print(f"Foot height range: {np.min(foot_z):.4f}m to {np.max(foot_z):.4f}m")
print(f"Sand surface: Z=0.501m")
print()

if np.mean(contacts) < 1:
    print("⚠️  WARNING: Very few contacts with sand!")
    print("   Robot is walking above sand, not pushing on it")
    print("   Need to:")
    print("   - Lower robot hip MORE")
    print("   - Increase amplitude MORE")
    print("   - Add downward force component")
else:
    print(f"✓ Good contact with sand ({np.mean(contacts):.1f} contacts/frame)")
