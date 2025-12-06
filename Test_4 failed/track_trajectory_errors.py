#!/usr/bin/env python3
"""
Track actual robot motion vs planned trajectory
Compare foot positions and visualize tracking error
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt
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
print("TRAJECTORY TRACKING ANALYSIS")
print("="*80)

# Load model
model = load_model_with_assets()
data = mujoco.MjData(model)

# Load trajectories
base_traj = np.load("base_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

# Load IK solutions
qpos_solutions = np.load("joint_solutions_ik.npy") if Path("joint_solutions_ik.npy").exists() else None

if qpos_solutions is None:
    print("[ERROR] Need to run ik_and_view.py first to generate joint_solutions_ik.npy")
    exit(1)

# Get site/body IDs
foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id
base_body_id = model.body(name='hip').id

print(f"\nAnalyzing {len(qpos_solutions)} trajectory points...")

# Track actual vs planned
actual_base = []
actual_foot1 = []
actual_foot2 = []

base_errors = []
foot1_errors = []
foot2_errors = []

for step, qpos in enumerate(qpos_solutions):
    # Forward kinematics
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    # Get actual positions
    base_pos = data.xpos[base_body_id].copy()
    foot1_pos = data.site_xpos[foot1_site_id].copy()
    foot2_pos = data.site_xpos[foot2_site_id].copy()
    
    actual_base.append(base_pos)
    actual_foot1.append(foot1_pos)
    actual_foot2.append(foot2_pos)
    
    # Compute errors
    base_err = np.linalg.norm(base_traj[step] - base_pos)
    foot1_err = np.linalg.norm(foot1_traj[step] - foot1_pos)
    foot2_err = np.linalg.norm(foot2_traj[step] - foot2_pos)
    
    base_errors.append(base_err)
    foot1_errors.append(foot1_err)
    foot2_errors.append(foot2_err)

actual_base = np.array(actual_base)
actual_foot1 = np.array(actual_foot1)
actual_foot2 = np.array(actual_foot2)

base_errors = np.array(base_errors)
foot1_errors = np.array(foot1_errors)
foot2_errors = np.array(foot2_errors)

# Print statistics
print("\n" + "="*80)
print("TRACKING ERROR STATISTICS")
print("="*80)

print(f"\nBase (Hip) Tracking:")
print(f"  Planned range: X=[{base_traj[:,0].min():.4f}, {base_traj[:,0].max():.4f}]")
print(f"  Actual range:  X=[{actual_base[:,0].min():.4f}, {actual_base[:,0].max():.4f}]")
print(f"  Planned range: Y=[{base_traj[:,1].min():.4f}, {base_traj[:,1].max():.4f}]")
print(f"  Actual range:  Y=[{actual_base[:,1].min():.4f}, {actual_base[:,1].max():.4f}]")
print(f"  Planned range: Z=[{base_traj[:,2].min():.4f}, {base_traj[:,2].max():.4f}]")
print(f"  Actual range:  Z=[{actual_base[:,2].min():.4f}, {actual_base[:,2].max():.4f}]")
print(f"  Error - Mean: {base_errors.mean():.4f}m, Max: {base_errors.max():.4f}m, Std: {base_errors.std():.4f}m")

print(f"\nFoot1 (Left) Tracking:")
print(f"  Planned range: X=[{foot1_traj[:,0].min():.4f}, {foot1_traj[:,0].max():.4f}]")
print(f"  Actual range:  X=[{actual_foot1[:,0].min():.4f}, {actual_foot1[:,0].max():.4f}]")
print(f"  Planned range: Y=[{foot1_traj[:,1].min():.4f}, {foot1_traj[:,1].max():.4f}]")
print(f"  Actual range:  Y=[{actual_foot1[:,1].min():.4f}, {actual_foot1[:,1].max():.4f}]")
print(f"  Planned range: Z=[{foot1_traj[:,2].min():.4f}, {foot1_traj[:,2].max():.4f}]")
print(f"  Actual range:  Z=[{actual_foot1[:,2].min():.4f}, {actual_foot1[:,2].max():.4f}]")
print(f"  Error - Mean: {foot1_errors.mean():.4f}m, Max: {foot1_errors.max():.4f}m, Std: {foot1_errors.std():.4f}m")

print(f"\nFoot2 (Right) Tracking:")
print(f"  Planned range: X=[{foot2_traj[:,0].min():.4f}, {foot2_traj[:,0].max():.4f}]")
print(f"  Actual range:  X=[{actual_foot2[:,0].min():.4f}, {actual_foot2[:,0].max():.4f}]")
print(f"  Planned range: Y=[{foot2_traj[:,1].min():.4f}, {foot2_traj[:,1].max():.4f}]")
print(f"  Actual range:  Y=[{actual_foot2[:,1].min():.4f}, {actual_foot2[:,1].max():.4f}]")
print(f"  Planned range: Z=[{foot2_traj[:,2].min():.4f}, {foot2_traj[:,2].max():.4f}]")
print(f"  Actual range:  Z=[{actual_foot2[:,2].min():.4f}, {actual_foot2[:,2].max():.4f}]")
print(f"  Error - Mean: {foot2_errors.mean():.4f}m, Max: {foot2_errors.max():.4f}m, Std: {foot2_errors.std():.4f}m")

# Visualize
print("\nGenerating comparison plots...")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle("Trajectory Tracking: Planned vs Actual", fontsize=14)

# Base trajectories
axes[0, 0].plot(base_traj[:, 0], 'b-', label='Planned', linewidth=2)
axes[0, 0].plot(actual_base[:, 0], 'r--', label='Actual', linewidth=2)
axes[0, 0].set_ylabel("Base X (m)")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(base_traj[:, 1], 'b-', label='Planned', linewidth=2)
axes[0, 1].plot(actual_base[:, 1], 'r--', label='Actual', linewidth=2)
axes[0, 1].set_ylabel("Base Y (m)")
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[0, 2].plot(base_traj[:, 2], 'b-', label='Planned', linewidth=2)
axes[0, 2].plot(actual_base[:, 2], 'r--', label='Actual', linewidth=2)
axes[0, 2].set_ylabel("Base Z (m)")
axes[0, 2].legend()
axes[0, 2].grid(True)

axes[0, 3].plot(base_errors, 'k-', linewidth=2)
axes[0, 3].set_ylabel("Base Error (m)")
axes[0, 3].set_title(f"Mean: {base_errors.mean():.4f}m")
axes[0, 3].grid(True)

# Foot1 trajectories
axes[1, 0].plot(foot1_traj[:, 0], 'b-', label='Planned', linewidth=2)
axes[1, 0].plot(actual_foot1[:, 0], 'r--', label='Actual', linewidth=2)
axes[1, 0].set_ylabel("Foot1 X (m)")
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(foot1_traj[:, 1], 'b-', label='Planned', linewidth=2)
axes[1, 1].plot(actual_foot1[:, 1], 'r--', label='Actual', linewidth=2)
axes[1, 1].set_ylabel("Foot1 Y (m)")
axes[1, 1].legend()
axes[1, 1].grid(True)

axes[1, 2].plot(foot1_traj[:, 2], 'b-', label='Planned', linewidth=2)
axes[1, 2].plot(actual_foot1[:, 2], 'r--', label='Actual', linewidth=2)
axes[1, 2].set_ylabel("Foot1 Z (m)")
axes[1, 2].legend()
axes[1, 2].grid(True)

axes[1, 3].plot(foot1_errors, 'k-', linewidth=2)
axes[1, 3].set_ylabel("Foot1 Error (m)")
axes[1, 3].set_title(f"Mean: {foot1_errors.mean():.4f}m")
axes[1, 3].grid(True)

# Foot2 trajectories
axes[2, 0].plot(foot2_traj[:, 0], 'b-', label='Planned', linewidth=2)
axes[2, 0].plot(actual_foot2[:, 0], 'r--', label='Actual', linewidth=2)
axes[2, 0].set_ylabel("Foot2 X (m)")
axes[2, 0].set_xlabel("Step #")
axes[2, 0].legend()
axes[2, 0].grid(True)

axes[2, 1].plot(foot2_traj[:, 1], 'b-', label='Planned', linewidth=2)
axes[2, 1].plot(actual_foot2[:, 1], 'r--', label='Actual', linewidth=2)
axes[2, 1].set_ylabel("Foot2 Y (m)")
axes[2, 1].set_xlabel("Step #")
axes[2, 1].legend()
axes[2, 1].grid(True)

axes[2, 2].plot(foot2_traj[:, 2], 'b-', label='Planned', linewidth=2)
axes[2, 2].plot(actual_foot2[:, 2], 'r--', label='Actual', linewidth=2)
axes[2, 2].set_ylabel("Foot2 Z (m)")
axes[2, 2].set_xlabel("Step #")
axes[2, 2].legend()
axes[2, 2].grid(True)

axes[2, 3].plot(foot2_errors, 'k-', linewidth=2)
axes[2, 3].set_ylabel("Foot2 Error (m)")
axes[2, 3].set_xlabel("Step #")
axes[2, 3].set_title(f"Mean: {foot2_errors.mean():.4f}m")
axes[2, 3].grid(True)

plt.tight_layout()
plt.savefig("trajectory_tracking_analysis.png", dpi=150, bbox_inches='tight')
print("  [OK] Saved trajectory_tracking_analysis.png")
plt.close()

# 3D trajectory comparison
fig = plt.figure(figsize=(14, 6))

# Base 3D
ax = fig.add_subplot(131, projection='3d')
ax.plot(base_traj[:, 0], base_traj[:, 1], base_traj[:, 2], 'b-', label='Planned', linewidth=2)
ax.plot(actual_base[:, 0], actual_base[:, 1], actual_base[:, 2], 'r--', label='Actual', linewidth=2)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Base (Hip) Trajectory")
ax.legend()

# Foot1 3D
ax = fig.add_subplot(132, projection='3d')
ax.plot(foot1_traj[:, 0], foot1_traj[:, 1], foot1_traj[:, 2], 'b-', label='Planned', linewidth=2)
ax.plot(actual_foot1[:, 0], actual_foot1[:, 1], actual_foot1[:, 2], 'r--', label='Actual', linewidth=2)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Foot1 (Left) Trajectory")
ax.legend()

# Foot2 3D
ax = fig.add_subplot(133, projection='3d')
ax.plot(foot2_traj[:, 0], foot2_traj[:, 1], foot2_traj[:, 2], 'b-', label='Planned', linewidth=2)
ax.plot(actual_foot2[:, 0], actual_foot2[:, 1], actual_foot2[:, 2], 'r--', label='Actual', linewidth=2)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Foot2 (Right) Trajectory")
ax.legend()

plt.savefig("trajectory_3d_comparison.png", dpi=150, bbox_inches='tight')
print("  [OK] Saved trajectory_3d_comparison.png")
plt.close()

print("\n" + "="*80)
