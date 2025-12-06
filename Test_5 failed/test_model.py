#!/usr/bin/env python3
"""
Quick model inspection and test
"""

import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
data = mujoco.MjData(model)

# Test some configurations
print("\n" + "="*80)
print("MODEL CONFIGURATION TEST")
print("="*80)

configs = [
    ([0, 0, 0], "All zeros"),
    ([0, -np.pi/4, np.pi/4], "Walking stance"),
    ([0, -np.pi/2, np.pi/2], "Flex stance"),
    ([-np.pi/4, -np.pi/4, np.pi/4], "Hip rotate + flex"),
]

for config, desc in configs:
    data.qpos[3:6] = config
    data.qpos[6:9] = config
    mujoco.mj_forward(model, data)
    
    left = data.site_xpos[0]
    right = data.site_xpos[1]
    
    print(f"\n{desc}:")
    print(f"  Left foot:  ({left[0]*1000:7.2f}, {left[1]*1000:7.2f}, {left[2]*1000:7.2f}) mm")
    print(f"  Right foot: ({right[0]*1000:7.2f}, {right[1]*1000:7.2f}, {right[2]*1000:7.2f}) mm")

print("\n" + "="*80)
print("\nConclusion: The robot model has feet positioned ~400mm above origin")
print("Need to generate trajectories relative to this reference frame")
print("="*80 + "\n")
