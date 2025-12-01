#!/usr/bin/env python3
"""
Find the actual reachable workspace of the feet
"""
import numpy as np
import mujoco
import os

# Load the model
script_dir = os.path.dirname(os.path.abspath(__file__))
mjcf_path = os.path.join(script_dir, "legged_robot_ik.xml")

try:
    model = mujoco.MjModel.from_xml_path(mjcf_path)
except:
    model = mujoco.load_model_from_xml(open(mjcf_path).read())

data = mujoco.MjData(model)

print("=== FINDING REACHABLE WORKSPACE ===\n")

# Test different joint configurations
best_z_foot1 = []
best_z_foot2 = []

# Test hip and ankle angles
for hip_angle in np.linspace(-np.pi/2, np.pi/2, 20):
    for ankle_angle in np.linspace(-np.pi/2, np.pi/2, 20):
        for knee_angle in np.linspace(-np.pi, -np.pi/3, 30):
            # Set floating base to reasonable position
            data.qpos[0:6] = [0, 0, 0.2, 1, 0, 0]  # x, y, z, quat (w last)
            
            # Left leg
            data.qpos[6] = hip_angle      # hip
            data.qpos[7] = knee_angle     # knee
            data.qpos[8] = ankle_angle    # ankle
            
            # Right leg
            data.qpos[9] = hip_angle      # hip
            data.qpos[10] = knee_angle    # knee
            data.qpos[11] = ankle_angle   # ankle
            
            mujoco.mj_kinematics(model, data)
            
            foot1_z = data.site_xpos[1][2]  # site_1 is foot1
            foot2_z = data.site_xpos[2][2]  # site_2 is foot2
            
            best_z_foot1.append(foot1_z)
            best_z_foot2.append(foot2_z)

best_z_foot1 = np.array(best_z_foot1)
best_z_foot2 = np.array(best_z_foot2)

print(f"Foot 1 (left leg) reachable Z range:")
print(f"  Min Z: {best_z_foot1.min():.6f} m")
print(f"  Max Z: {best_z_foot1.max():.6f} m")
print(f"  Mean Z: {best_z_foot1.mean():.6f} m")
print(f"  Std dev: {best_z_foot1.std():.6f} m")

print(f"\nFoot 2 (right leg) reachable Z range:")
print(f"  Min Z: {best_z_foot2.min():.6f} m")
print(f"  Max Z: {best_z_foot2.max():.6f} m")
print(f"  Mean Z: {best_z_foot2.mean():.6f} m")
print(f"  Std dev: {best_z_foot2.std():.6f} m")

print(f"\nCombined reachable Z range: {min(best_z_foot1.min(), best_z_foot2.min()):.6f} to {max(best_z_foot1.max(), best_z_foot2.max()):.6f} m")

print("\n=== RECOMMENDED TRAJECTORY ADJUSTMENT ===")
min_z = min(best_z_foot1.min(), best_z_foot2.min())
max_z = max(best_z_foot1.max(), best_z_foot2.max())
mid_z = (min_z + max_z) / 2
swing_height = max_z - min_z

print(f"Stance phase Z (lowest point): {min_z:.6f} m")
print(f"Swing phase max Z (highest point): {max_z:.6f} m")
print(f"Available clearance for swing: {swing_height:.6f} m")
print(f"Recommended COM height: {mid_z + 0.01:.6f} m")
print(f"\nScale trajectories by shifting Z to:")
print(f"  Stance: {min_z:.6f} m (instead of 0.0)")
print(f"  Swing: {max_z - 0.005:.6f} m (or use any value in valid range)")
