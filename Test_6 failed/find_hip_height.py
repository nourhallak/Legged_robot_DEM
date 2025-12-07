#!/usr/bin/env python3
"""Find the hip height that makes feet rest on sand"""

import mujoco
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path("legged_robot_sand.xml")
data = mujoco.MjData(model)

# Move hip down in Z to make feet touch sand
hip_body_id = model.body("hip").id
foot1_id = model.body("foot_1").id
foot2_id = model.body("foot_2").id

sand_top_z = 0.444  # Top layer of sand + sphere radius (0.442 + 0.002)

print(f"\nFinding hip height for feet to rest on sand (Z={sand_top_z:.3f}m)\n")

# Test various hip heights
for hip_z in np.arange(0.40, 0.50, 0.01):
    data.qpos[0:3] = [0, 0, hip_z]  # Set base X, Y, Z
    data.qpos[3:] = 0  # All joint angles to zero
    
    mujoco.mj_forward(model, data)
    
    foot1_z = data.xpos[foot1_id][2]
    foot2_z = data.xpos[foot2_id][2]
    
    print(f"Hip Z={hip_z:.3f} → Foot_1 Z={foot1_z:.4f}, Foot_2 Z={foot2_z:.4f} (diff={sand_top_z-foot1_z:+.4f})")
    
    if abs(foot1_z - sand_top_z) < 0.001:
        print(f"\n✓ FOUND: Hip Z={hip_z:.3f} makes feet rest on sand")
        break
