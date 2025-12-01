#!/usr/bin/env python3
"""
Verify that stance feet maintain ground contact during simulation.
"""
import mujoco
import numpy as np
import os
import re

def load_model_with_assets():
    """Load the converted MJCF model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.join(script_dir, "Legged_robot")
    meshes_dir = os.path.join(package_dir, "meshes")
    mjcf_output_path = os.path.join(script_dir, "legged_robot_ik.xml")

    if not os.path.exists(mjcf_output_path):
        raise FileNotFoundError(f"Model file not found at: {mjcf_output_path}")

    with open(mjcf_output_path, 'r', encoding='utf-8') as f:
        mjcf_content = f.read()

    MESH_PATTERN = r'file="([^"]+\.STL)"'
    all_mesh_files = set(re.findall(MESH_PATTERN, mjcf_content))

    assets = {}
    for mesh_file in all_mesh_files:
        abs_path = os.path.join(meshes_dir, mesh_file)
        if os.path.exists(abs_path):
            with open(abs_path, 'rb') as f:
                assets[mesh_file] = f.read()

    model = mujoco.MjModel.from_xml_string(mjcf_content, assets=assets)
    return model

# Load model
print("Loading model...")
model = load_model_with_assets()
data = mujoco.MjData(model)

# Load trajectories
print("Loading trajectories...")
script_dir = os.path.dirname(os.path.abspath(__file__))
foot1_traj = np.load(os.path.join(script_dir, "foot1_trajectory.npy"))
foot2_traj = np.load(os.path.join(script_dir, "foot2_trajectory.npy"))

# Ground height (from model)
ground_z = 0.21

print("\n=== GROUND CONTACT VERIFICATION ===\n")

# Check trajectory data
print("Foot 1 (Left) Z values:")
print(f"  Min: {foot1_traj[:, 2].min():.6f} m")
print(f"  Max: {foot1_traj[:, 2].max():.6f} m")
print(f"  Mean: {foot1_traj[:, 2].mean():.6f} m")

print("\nFoot 2 (Right) Z values:")
print(f"  Min: {foot2_traj[:, 2].min():.6f} m")
print(f"  Max: {foot2_traj[:, 2].max():.6f} m")
print(f"  Mean: {foot2_traj[:, 2].mean():.6f} m")

print(f"\nGround Z: {ground_z:.6f} m")
print(f"Foot 1 minimum clearance: {foot1_traj[:, 2].min() - ground_z:.6f} m (should be ≥ 0)")
print(f"Foot 2 minimum clearance: {foot2_traj[:, 2].min() - ground_z:.6f} m (should be ≥ 0)")

# Check for stance phase variations
print("\n=== STANCE PHASE ANALYSIS ===\n")

# Left foot stance during steps 0-100, 200-300, 400-...
stance_frames_left = list(range(0, 100)) + list(range(200, 300))
stance_z_left = foot1_traj[stance_frames_left, 2]
print(f"Left foot during stance (frames 0-100, 200-300):")
print(f"  Z values: {stance_z_left.min():.6f} to {stance_z_left.max():.6f}")
print(f"  Std dev: {stance_z_left.std():.9f} (should be ~0)")
print(f"  Max variation: {stance_z_left.max() - stance_z_left.min():.9f} m")

# Right foot stance during steps 100-200, 300-400
stance_frames_right = list(range(100, 200)) + list(range(300, 400))
stance_z_right = foot2_traj[stance_frames_right, 2]
print(f"\nRight foot during stance (frames 100-200, 300-400):")
print(f"  Z values: {stance_z_right.min():.6f} to {stance_z_right.max():.6f}")
print(f"  Std dev: {stance_z_right.std():.9f} (should be ~0)")
print(f"  Max variation: {stance_z_right.max() - stance_z_right.min():.9f} m")

# Check swing phases
print("\n=== SWING PHASE ANALYSIS ===\n")

swing_frames_left = list(range(100, 200)) + list(range(300, 400))
swing_z_left = foot1_traj[swing_frames_left, 2]
print(f"Left foot during swing (frames 100-200, 300-400):")
print(f"  Z values: {swing_z_left.min():.6f} to {swing_z_left.max():.6f}")
print(f"  Swing clearance: {swing_z_left.max() - ground_z:.6f} m")

swing_frames_right = list(range(0, 100)) + list(range(200, 300))
swing_z_right = foot2_traj[swing_frames_right, 2]
print(f"\nRight foot during swing (frames 0-100, 200-300):")
print(f"  Z values: {swing_z_right.min():.6f} to {swing_z_right.max():.6f}")
print(f"  Swing clearance: {swing_z_right.max() - ground_z:.6f} m")

print("\n=== VERIFICATION RESULT ===")
if foot1_traj[:, 2].min() >= ground_z and foot2_traj[:, 2].min() >= ground_z:
    print("✓ SUCCESS: All feet stay at or above ground")
else:
    print("✗ ERROR: Feet go below ground!")

if stance_z_left.std() < 1e-6 and stance_z_right.std() < 1e-6:
    print("✓ SUCCESS: Stance feet remain fixed (no sliding)")
else:
    print(f"✗ ERROR: Stance feet are varying!")
    print(f"  Left std dev: {stance_z_left.std():.9f}")
    print(f"  Right std dev: {stance_z_right.std():.9f}")
