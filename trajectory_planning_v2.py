#!/usr/bin/env python3
"""
Generate walking trajectories with new kinematic constraints.
Uses updated knee limits (-120 to +60 degrees) for realistic gait.
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

print("=" * 80)
print("FINDING REACHABLE Z RANGE WITH NEW KNEE LIMITS (-120 to +60 degrees)")
print("=" * 80)

# Test the new limits to find the actual reachable Z range
hip_angle = 0.0
ankle_angle = 0.0

# New limits from analysis
knee_min = -2.0944  # -120 degrees
knee_max = 1.0472   # +60 degrees
knee_angles_rad = np.linspace(knee_min, knee_max, 200)

z_positions = []

for knee_angle in knee_angles_rad:
    data.qpos[0:6] = [0, 0, 0.2, 1, 0, 0]
    data.qpos[6] = hip_angle
    data.qpos[7] = knee_angle
    data.qpos[8] = ankle_angle
    data.qpos[9] = hip_angle
    data.qpos[10] = knee_angle
    data.qpos[11] = ankle_angle
    
    mujoco.mj_kinematics(model, data)
    
    foot1_z = data.site_xpos[1][2]
    z_positions.append(foot1_z)

z_positions = np.array(z_positions)
z_min = z_positions.min()
z_max = z_positions.max()

print(f"\nWith new limits (-120 to +60 degrees):")
print(f"  Reachable Z range: {z_min:.6f} to {z_max:.6f} m")
print(f"  Available range: {z_max - z_min:.6f} m ({(z_max - z_min)*1000:.2f} mm)")

# Find the angle that gives minimum Z (good for stance)
min_idx = np.argmin(z_positions)
stance_angle = np.degrees(knee_angles_rad[min_idx])
print(f"  Best stance angle: {stance_angle:.1f}°")

print("\n" + "=" * 80)
print("GENERATING WALKING TRAJECTORIES")
print("=" * 80)

num_steps = 400
stride_length = 0.16

# Create time vector
t = np.linspace(0, 1, num_steps)

# --- COM TRAJECTORY ---
com_x = stride_length * t
com_y = 0.005 * np.sin(4 * np.pi * t)
com_bounce = 0.01  # Adjusted for new Z range
com_z_base = z_min + (z_max - z_min) * 0.5 + 0.005  # Middle of range + offset
com_z = com_z_base + com_bounce * np.cos(4 * np.pi * t)

com_trajectory = np.column_stack([com_x, com_y, com_z])

# --- FOOT 1 TRAJECTORY (Left leg) ---
foot1_phase = (t * 2) % 2

foot1_x = stride_length * t
foot1_y = -0.01 * np.ones(num_steps)
foot1_z = np.zeros(num_steps)

# Swing and stance phases
for i in range(num_steps):
    phase = foot1_phase[i]
    if phase <= 1.0:  # Swing phase (0-1)
        swing_fraction = phase
        # Parabolic swing trajectory
        swing_lift = (z_max - z_min) * 0.6 * np.sin(np.pi * swing_fraction)
        foot1_z[i] = z_min + swing_lift
    else:  # Stance phase (1-2)
        foot1_z[i] = z_min

foot1_trajectory = np.column_stack([foot1_x, foot1_y, foot1_z])

# --- FOOT 2 TRAJECTORY (Right leg) ---
foot2_phase = ((t * 2 + 1) % 2)

foot2_x = stride_length * t
foot2_y = 0.01 * np.ones(num_steps)
foot2_z = np.zeros(num_steps)

for i in range(num_steps):
    phase = foot2_phase[i]
    if phase <= 1.0:  # Swing phase
        swing_fraction = phase
        swing_lift = (z_max - z_min) * 0.6 * np.sin(np.pi * swing_fraction)
        foot2_z[i] = z_min + swing_lift
    else:  # Stance phase
        foot2_z[i] = z_min

foot2_trajectory = np.column_stack([foot2_x, foot2_y, foot2_z])

# --- VERIFY TRAJECTORIES ---
print("\n=== TRAJECTORY VERIFICATION ===\n")

print("COM Trajectory:")
print(f"  X: {com_trajectory[:, 0].min():.4f} to {com_trajectory[:, 0].max():.4f}")
print(f"  Y: {com_trajectory[:, 1].min():.4f} to {com_trajectory[:, 1].max():.4f}")
print(f"  Z: {com_trajectory[:, 2].min():.6f} to {com_trajectory[:, 2].max():.6f}")

print("\nFoot 1 Trajectory:")
print(f"  X: {foot1_trajectory[:, 0].min():.4f} to {foot1_trajectory[:, 0].max():.4f}")
print(f"  Y: {foot1_trajectory[:, 1].min():.4f} to {foot1_trajectory[:, 1].max():.4f}")
print(f"  Z: {foot1_trajectory[:, 2].min():.6f} to {foot1_trajectory[:, 2].max():.6f}")
print(f"  Swing height: {foot1_trajectory[:, 2].max() - foot1_trajectory[:, 2].min():.6f} m")

print("\nFoot 2 Trajectory:")
print(f"  X: {foot2_trajectory[:, 0].min():.4f} to {foot2_trajectory[:, 0].max():.4f}")
print(f"  Y: {foot2_trajectory[:, 1].min():.4f} to {foot2_trajectory[:, 1].max():.4f}")
print(f"  Z: {foot2_trajectory[:, 2].min():.6f} to {foot2_trajectory[:, 2].max():.6f}")
print(f"  Swing height: {foot2_trajectory[:, 2].max() - foot2_trajectory[:, 2].min():.6f} m")

print("\n=== SAMPLE FRAMES ===\n")
print("Frame  | COM X   | COM Z   | Foot1 Z | Foot2 Z | Description")
print("-------|---------|---------|---------|---------|------------------")
for i in [0, 50, 100, 150, 200, 250, 300, 350, 399]:
    phase1 = (i / num_steps * 2) % 2
    phase2 = ((i / num_steps * 2 + 1) % 2)
    
    desc1 = "Swing" if phase1 <= 1.0 else "Stance"
    desc2 = "Swing" if phase2 <= 1.0 else "Stance"
    desc = f"F1:{desc1} F2:{desc2}"
    
    print(f"{i:5d}  | {com_trajectory[i, 0]:.4f} | {com_trajectory[i, 2]:.6f} | {foot1_trajectory[i, 2]:.6f}  | {foot2_trajectory[i, 2]:.6f}  | {desc}")

# --- SAVE TRAJECTORIES ---
print("\n")
np.save("com_trajectory.npy", com_trajectory)
np.save("foot1_trajectory.npy", foot1_trajectory)
np.save("foot2_trajectory.npy", foot2_trajectory)

print("OK Trajectories saved:")
print("   - com_trajectory.npy")
print("   - foot1_trajectory.npy")
print("   - foot2_trajectory.npy")

print("\nThese trajectories represent realistic walking with:")
print("   - New knee limits: -120 to +60 degrees")
print(f"   - Foot stance Z: {z_min:.6f} m")
print(f"   - Foot swing Z: {z_max:.6f} m")
print("   - Alternating foot phases")
print("   - COM vertical oscillation")
print("   - Proper forward progression")
