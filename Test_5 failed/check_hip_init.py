#!/usr/bin/env python3
"""Quick check of initial hip position"""

import mujoco

model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
data = mujoco.MjData(model)

print("\nInitial state:")
print(f"Hip body position: {data.xpos[1]*1000}")  # Body 1 is hip
print(f"qpos: {data.qpos}")
print(f"\nHip height (Z): {data.xpos[1][2]*1000:.1f} mm")
print(f"Hip Y: {data.xpos[1][1]*1000:.1f} mm")

