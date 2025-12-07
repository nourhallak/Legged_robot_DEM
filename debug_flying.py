#!/usr/bin/env python3
"""Debug: Check why robot is flying instead of walking."""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

print("DIAGNOSIS: Robot Flying Issues")
print("="*70)

# Test with WEAK control first
frequency = 0.5
amplitude_test = 0.2  # Much weaker
num_actuators = len(data.ctrl)
dt = model.opt.timestep

foot_min_z = []
foot_max_z = []
contact_count = []
hip_z_list = []

print(f"Testing with amplitude={amplitude_test}...")

for step in range(10000):  # 20 seconds
    t = step * dt
    
    phase = np.sin(2.0 * np.pi * frequency * t)
    phase_offset = np.sin(2.0 * np.pi * frequency * (t + 0.5/frequency))
    
    for i in range(num_actuators):
        if i < num_actuators // 2:
            data.ctrl[i] = -amplitude_test * max(phase, 0)
        else:
            data.ctrl[i] = -amplitude_test * max(phase_offset, 0)
    
    mujoco.mj_step(model, data)
    
    if step % 100 == 0:
        foot1_z = data.xpos[model.body('foot_1').id][2]
        foot2_z = data.xpos[model.body('foot_2').id][2]
        hip_z = data.xpos[model.body('hip').id][2]
        
        foot_min_z.append(min(foot1_z, foot2_z))
        foot_max_z.append(max(foot1_z, foot2_z))
        contact_count.append(data.ncon)
        hip_z_list.append(hip_z)

print("\nRESULTS:")
print(f"  Foot height: {np.min(foot_min_z):.4f}m to {np.max(foot_max_z):.4f}m")
print(f"  Hip height: {np.min(hip_z_list):.4f}m to {np.max(hip_z_list):.4f}m")
print(f"  Sand surface: Z=0.501m (top layer)")
print(f"  Floor: Z=0.483m (bottom)")
print(f"  Contacts: {np.mean(contact_count):.1f} per frame avg")
print()

if np.max(hip_z_list) > 0.5:
    print("⚠️  ROBOT IS FLYING UP!")
    print("   Cause: Control amplitude too strong")
    print("   Solution: Reduce amplitude further OR use gravity-based gait")
    print()
    print(f"   Recommended amplitude: 0.1 to 0.15 rad")
else:
    print("✓ Robot staying low (good)")
    
print("\nTip: Use GRAVITY to keep robot on ground")
print("Add downward bias force OR reduce control strength")
