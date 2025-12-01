#!/usr/bin/env python3
"""
Analyze actual leg dimensions from the URDF to generate kinematically-accurate trajectories
"""
import numpy as np
import mujoco
import mujoco as mj
import os

# Load model
script_dir = os.path.dirname(os.path.abspath(__file__))
mjcf_path = os.path.join(script_dir, "legged_robot_ik.xml")

try:
    model = mujoco.MjModel.from_xml_path(mjcf_path)
except:
    model = mujoco.load_model_from_xml(open(mjcf_path).read())

data = mujoco.MjData(model)

print("=" * 80)
print("LEG DIMENSION ANALYSIS")
print("=" * 80)

# Get body positions to measure leg segments
print("\nBody positions in structure:")
for i in range(model.nbody):
    body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
    parent_id = model.body_parentid[i]
    parent_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, parent_id)
    print(f"  {body_name} (parent: {parent_name})")

# Measure leg lengths by computing forward kinematics
print("\n" + "=" * 80)
print("LEG SEGMENT MEASUREMENTS")
print("=" * 80)

# Set hip in neutral position
data.qpos[0:6] = [0, 0, 0.2, 1, 0, 0]  # Position and orientation
data.qpos[6] = 0    # Hip L
data.qpos[7] = 0    # Knee L
data.qpos[8] = 0    # Ankle L

mujoco.mj_kinematics(model, data)

# Get positions of key bodies
hip_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "hip")
link_2_1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link_2_1")
link_1_1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link_1_1")
foot_1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "foot_1")

# Foot site positions
foot1_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "foot1_site")

hip_pos = data.xpos[hip_id]
link_2_1_pos = data.xpos[link_2_1_id]
link_1_1_pos = data.xpos[link_1_1_id]
foot_1_pos = data.xpos[foot_1_id]
foot1_site_pos = data.site_xpos[foot1_site_id]

# Calculate segment lengths
hip_to_thigh = np.linalg.norm(link_2_1_pos - hip_pos)
thigh_to_knee = np.linalg.norm(link_1_1_pos - link_2_1_pos)
knee_to_foot = np.linalg.norm(foot_1_pos - link_1_1_pos)
hip_to_foot = np.linalg.norm(foot1_site_pos - hip_pos)

print(f"\nLeft leg segments (neutral position):")
print(f"  Hip to Thigh (link_2_1): {hip_to_thigh:.6f} m")
print(f"  Thigh to Knee (link_1_1): {thigh_to_knee:.6f} m")
print(f"  Knee to Foot: {knee_to_foot:.6f} m")
print(f"  Total leg length (hip to foot): {hip_to_foot:.6f} m")

# Test different knee angles to find swing height and reach
print("\n" + "=" * 80)
print("KNEE ANGLE EFFECTS ON REACH")
print("=" * 80)

z_values = []
for knee_angle in np.linspace(-2.0944, 1.0472, 20):  # -120 to +60 degrees
    data.qpos[6] = 0    # Hip L
    data.qpos[7] = knee_angle  # Knee L
    data.qpos[8] = 0    # Ankle L
    
    mujoco.mj_kinematics(model, data)
    foot1_site_pos = data.site_xpos[foot1_site_id]
    z_values.append(foot1_site_pos[2])

z_values = np.array(z_values)
print(f"\nZ position range with different knee angles:")
print(f"  Min Z (most extended): {z_values.min():.6f} m")
print(f"  Max Z (most flexed): {z_values.max():.6f} m")
print(f"  Available vertical range: {z_values.max() - z_values.min():.6f} m")

# Recommend trajectory parameters
print("\n" + "=" * 80)
print("RECOMMENDED TRAJECTORY PARAMETERS")
print("=" * 80)

foot_z_min = z_values.min()
foot_z_max = z_values.max()
swing_clearance = (foot_z_max - foot_z_min) * 0.4  # Use 40% of available range

print(f"\nStance phase Z (foot on ground): {foot_z_min:.6f} m")
print(f"Swing phase Z (maximum lift): {foot_z_min + swing_clearance:.6f} m")
print(f"Available swing clearance: {swing_clearance:.6f} m ({swing_clearance*1000:.2f} mm)")

# COM should be above the legs for stability
com_z_natural = foot_z_min + (hip_to_foot * 0.5)  # COM roughly at midpoint of leg
print(f"\nNatural COM height (leg midpoint): {com_z_natural:.6f} m")

print(f"\nSuggested trajectory Z values:")
print(f"  COM Z: {com_z_natural:.6f} m")
print(f"  Stance Z: {foot_z_min:.6f} m")
print(f"  Swing peak Z: {foot_z_min + swing_clearance:.6f} m")

print("\n" + "=" * 80)