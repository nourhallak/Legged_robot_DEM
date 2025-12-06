#!/usr/bin/env python3
"""
Debug IK joint angles to see if they enable lifting
"""

import numpy as np
import mujoco
import re
from pathlib import Path

def load_model_with_assets(xml_path="legged_robot_ik.xml"):
    xml_file = Path(xml_path)
    if not xml_file.exists():
        xml_file = Path(".") / "legged_robot_ik.xml"
    
    with open(xml_file, "r") as f:
        xml_content = f.read()
    
    asset_dir = xml_file.parent / "Legged_robot" / "meshes"
    
    for match in re.finditer(r'filename="([^"]*\.stl)"', xml_content):
        stl_file = match.group(1)
        full_path = asset_dir / stl_file
        
        if full_path.exists():
            xml_content = xml_content.replace(f'filename="{stl_file}"', f'filename="{full_path}"')
    
    return mujoco.MjModel.from_xml_string(xml_content)

print("="*80)
print("IK JOINT ANGLE ANALYSIS")
print("="*80)

model = load_model_with_assets()
data = mujoco.MjData(model)

# Load trajectories
base_traj = np.load("base_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

# Load IK solutions
qpos_solutions = np.load("joint_solutions_ik.npy")

print(f"\nAnalyzing {len(qpos_solutions)} trajectory points...")

# Check joint ranges
print(f"\nJoint angle ranges from model:")
for j in range(model.nq):
    if j < model.jnt_range.shape[0]:
        jmin, jmax = model.jnt_range[j]
        print(f"  Joint {j}: [{jmin:.4f}, {jmax:.4f}] rad")

print(f"\nIK solution joint angles:")
foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id

# Sample key frames
key_frames = [0, 25, 50, 75]  # Early, mid-swing, late-swing, stance

for frame in key_frames:
    qpos = qpos_solutions[frame]
    base_t = base_traj[frame]
    foot1_t = foot1_traj[frame]
    foot2_t = foot2_traj[frame]
    
    # Forward kinematics
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    foot1_actual = data.site_xpos[foot1_site_id]
    foot2_actual = data.site_xpos[foot2_site_id]
    
    print(f"\n--- Frame {frame} (Step in gait cycle) ---")
    print(f"  Planned foot1 Z: {foot1_t[2]:.4f}m, Actual: {foot1_actual[2]:.4f}m")
    print(f"  Planned foot2 Z: {foot2_t[2]:.4f}m, Actual: {foot2_actual[2]:.4f}m")
    print(f"  Joint angles (deg):")
    print(f"    Leg1 (joints 3-5): [{np.degrees(qpos[3]):.1f}, {np.degrees(qpos[4]):.1f}, {np.degrees(qpos[5]):.1f}]")
    print(f"    Leg2 (joints 6-8): [{np.degrees(qpos[6]):.1f}, {np.degrees(qpos[7]):.1f}, {np.degrees(qpos[8]):.1f}]")

# Check if feet actually lift
print(f"\n" + "="*80)
print("FOOT HEIGHT ANALYSIS")
print("="*80)

foot1_z_min = np.min(foot1_traj[:, 2])
foot1_z_max = np.max(foot1_traj[:, 2])
foot2_z_min = np.min(foot2_traj[:, 2])
foot2_z_max = np.max(foot2_traj[:, 2])

print(f"\nPlanned foot heights:")
print(f"  Foot1 Z range: [{foot1_z_min:.4f}, {foot1_z_max:.4f}]m")
print(f"  Foot2 Z range: [{foot2_z_min:.4f}, {foot2_z_max:.4f}]m")
print(f"  Lift height (foot1): {foot1_z_max - foot1_z_min:.4f}m")
print(f"  Lift height (foot2): {foot2_z_max - foot2_z_min:.4f}m")

# Compute actual foot heights from IK solutions
actual_foot1_z = []
actual_foot2_z = []

for qpos in qpos_solutions:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    actual_foot1_z.append(data.site_xpos[foot1_site_id][2])
    actual_foot2_z.append(data.site_xpos[foot2_site_id][2])

actual_foot1_z = np.array(actual_foot1_z)
actual_foot2_z = np.array(actual_foot2_z)

print(f"\nActual foot heights from IK:")
print(f"  Foot1 Z range: [{actual_foot1_z.min():.4f}, {actual_foot1_z.max():.4f}]m")
print(f"  Foot2 Z range: [{actual_foot2_z.min():.4f}, {actual_foot2_z.max():.4f}]m")
print(f"  Actual lift height (foot1): {actual_foot1_z.max() - actual_foot1_z.min():.4f}m")
print(f"  Actual lift height (foot2): {actual_foot2_z.max() - actual_foot2_z.min():.4f}m")

# Check if lift is happening
if actual_foot1_z.max() - actual_foot1_z.min() < 0.001:
    print(f"\n[WARNING] Foot1 is NOT lifting! Height variation too small.")
if actual_foot2_z.max() - actual_foot2_z.min() < 0.001:
    print(f"[WARNING] Foot2 is NOT lifting! Height variation too small.")

print(f"\n" + "="*80)
