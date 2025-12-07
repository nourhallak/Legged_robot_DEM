#!/usr/bin/env python3
"""
Check sand and robot contact - simple version
"""

import mujoco
import numpy as np

print("="*70)
print("SAND CONTACT DIAGNOSTIC")
print("="*70)

model = mujoco.MjModel.from_xml_path('legged_robot_sand_top_surface_v2.xml')
data = mujoco.MjData(model)

# Initial state
hip_id = model.body('hip').id
foot1_id = model.body('foot_1').id
foot2_id = model.body('foot_2').id

print("\nINITIAL STATE:")
print(f"  Hip Z: {data.xpos[hip_id][2]:.4f}m (should be 0.52m)")
print(f"  Foot1 Z: {data.xpos[foot1_id][2]:.4f}m")
print(f"  Foot2 Z: {data.xpos[foot2_id][2]:.4f}m")
print(f"  Sand top layer: 0.438m")
print(f"  Floor: 0.420m")
print(f"  Initial contacts: {data.ncon}")

# Run 10 steps with control
print("\nRUNNING WITH CONTROL...")
for step in range(500):  # 1 second
    t = data.time
    
    # Trotting control
    phase = np.sin(2.0 * np.pi * 0.5 * t)
    phase_offset = np.sin(2.0 * np.pi * 0.5 * (t + 1.0))
    
    for i in range(model.nu):
        if i < 3:
            data.ctrl[i] = -0.2 * max(phase, 0)
        else:
            data.ctrl[i] = -0.2 * max(phase_offset, 0)
    
    mujoco.mj_step(model, data)

# After simulation
hip_pos = data.xpos[hip_id]
foot1_pos = data.xpos[foot1_id]
foot2_pos = data.xpos[foot2_id]

print("\nAFTER 1 SECOND:")
print(f"  Hip Z: {hip_pos[2]:.4f}m")
print(f"  Foot1 Z: {foot1_pos[2]:.4f}m")
print(f"  Foot2 Z: {foot2_pos[2]:.4f}m")
print(f"  Contacts: {data.ncon}")

# Check sand particles
print("\nSAND PARTICLE CHECK:")
# Count sand bodies
sand_count = 0
sand_z_values = []
for bid in range(model.nbody):
    bname = model.body(bid).name
    if bname.startswith('sand_'):
        sand_count += 1
        if sand_count <= 5:
            z = data.xpos[bid][2]
            sand_z_values.append(z)
            print(f"  {bname}: Z={z:.4f}m")

print(f"  Total sand particles: {sand_count}")

# Final verdict
print("\n" + "="*70)
print("VERDICT:")
print("="*70)

avg_foot_z = (foot1_pos[2] + foot2_pos[2]) / 2
print(f"\nAverage foot height: {avg_foot_z:.4f}m")
print(f"Sand top surface:    0.438m")
diff = avg_foot_z - 0.438

if diff > 0.05:
    print(f"\nPROBLEM: Feet are {diff:.4f}m ABOVE sand surface")
    print("         Robot is NOT touching sand")
    print("\nSOLUTION: Lower robot or raise sand")
elif diff > 0.01:
    print(f"\nWARNING: Feet are {diff:.4f}m above sand (marginal contact)")
elif diff < -0.01:
    print(f"\nINFO: Feet are {-diff:.4f}m below sand surface (good contact)")
else:
    print(f"\nOK: Feet near sand surface (distance: {diff:.4f}m)")

print(f"\nNumber of contacts: {data.ncon} (should be > 0 for sand contact)")
