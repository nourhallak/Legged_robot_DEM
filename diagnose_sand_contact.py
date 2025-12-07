#!/usr/bin/env python3
"""
Diagnostic script to check robot position relative to sand
"""

import mujoco
import numpy as np

print("Loading model...")
model = mujoco.MjModel.from_xml_path('legged_robot_sand_top_surface_v2.xml')
data = mujoco.MjData(model)

print("\n" + "="*70)
print("ROBOT AND SAND POSITION DIAGNOSTIC")
print("="*70)

# Get robot info
hip_id = model.body('hip').id
hip_pos = data.xpos[hip_id]
print(f"\nRobot Hip Position:")
print(f"  X: {hip_pos[0]:.4f}m")
print(f"  Y: {hip_pos[1]:.4f}m")
print(f"  Z: {hip_pos[2]:.4f}m")

# Get foot positions
foot1_id = model.body('foot_1').id
foot2_id = model.body('foot_2').id

foot1_pos = data.xpos[foot1_id]
foot2_pos = data.xpos[foot2_id]

print(f"\nFoot 1 Position:")
print(f"  X: {foot1_pos[0]:.4f}m")
print(f"  Y: {foot1_pos[1]:.4f}m")
print(f"  Z: {foot1_pos[2]:.4f}m")

print(f"\nFoot 2 Position:")
print(f"  X: {foot2_pos[0]:.4f}m")
print(f"  Y: {foot2_pos[1]:.4f}m")
print(f"  Z: {foot2_pos[2]:.4f}m")

# Get sand info (check first few sand particles)
print(f"\n\nSand Particle Positions (first 5):")
for i in range(5):
    body_name = f"sand_0_0_{i}"
    try:
        sand_id = model.body(body_name).id
        sand_pos = data.xpos[sand_id]
        print(f"  {body_name}: Z={sand_pos[2]:.4f}m")
    except:
        pass

# Find sand Z range
print(f"\nSand Layer Information:")
print(f"  Layer 0 (bottom): Z ≈ 0.426m")
print(f"  Layer 1 (middle): Z ≈ 0.432m")
print(f"  Layer 2 (top):    Z ≈ 0.438m")

print(f"\nFloor: Z = 0.420m")

print(f"\n" + "="*70)
print("ANALYSIS:")
print("="*70)

avg_foot_z = (foot1_pos[2] + foot2_pos[2]) / 2
print(f"\nAverage Foot Height: {avg_foot_z:.4f}m")
print(f"Sand Top Surface:    {0.438:.4f}m")

if avg_foot_z > 0.442:
    print(f"\n⚠ WARNING: Feet are ABOVE sand surface ({avg_foot_z - 0.438:.4f}m above)")
    print("           Robot NOT in contact with sand!")
elif avg_foot_z < 0.420:
    print(f"\n⚠ WARNING: Feet are BELOW floor ({0.420 - avg_foot_z:.4f}m below)")
else:
    print(f"\n✓ Feet are at sand surface level (good)")

print(f"\nHip Height: {hip_pos[2]:.4f}m")
print(f"Hip should be at: 0.520m (ON TOP)")

# Simulate one step to see what happens
print(f"\n" + "="*70)
print("RUNNING 1 STEP SIMULATION...")
print("="*70)

for step in range(100):  # 0.2 seconds
    # Apply weak trotting control
    t = data.time
    phase = np.sin(2.0 * np.pi * 0.5 * t)
    phase_offset = np.sin(2.0 * np.pi * 0.5 * (t + 1.0))
    
    for i in range(model.nu):
        if i < 3:
            data.ctrl[i] = -0.2 * max(phase, 0)
        else:
            data.ctrl[i] = -0.2 * max(phase_offset, 0)
    
    mujoco.mj_step(model, data)

# Check positions after simulation
hip_pos = data.xpos[hip_id]
foot1_pos = data.xpos[foot1_id]
foot2_pos = data.xpos[foot2_id]

print(f"\nAfter 0.2 seconds simulation:")
print(f"  Hip Z: {hip_pos[2]:.4f}m")
print(f"  Foot1 Z: {foot1_pos[2]:.4f}m")
print(f"  Foot2 Z: {foot2_pos[2]:.4f}m")
print(f"  Contacts: {data.ncon}")

avg_foot_z = (foot1_pos[2] + foot2_pos[2]) / 2
print(f"\nAverage Foot Height: {avg_foot_z:.4f}m")
print(f"Sand Top Surface:    0.438m")

if avg_foot_z > 0.442:
    print(f"\n⚠ PROBLEM: Feet are still ABOVE sand!")
    print("Solution: Need to lower robot or raise sand")
elif avg_foot_z >= 0.438:
    print(f"\n✓ Feet at/touching sand surface!")
else:
    print(f"\n✓ Feet below sand top (might be within sand)")
