#!/usr/bin/env python3
"""
Diagnose IK convergence issues by checking trajectory reachability
"""

import numpy as np
import mujoco
import re
from pathlib import Path

def load_model_with_assets(xml_path="legged_robot_ik.xml"):
    """Load MuJoCo model with correct asset paths"""
    
    xml_file = Path(xml_path)
    if not xml_file.exists():
        xml_file = Path(".") / "legged_robot_ik.xml"
    
    with open(xml_file, "r") as f:
        xml_content = f.read()
    
    # Update mesh paths
    asset_dir = xml_file.parent / "Legged_robot" / "meshes"
    
    for match in re.finditer(r'filename="([^"]*\.stl)"', xml_content):
        stl_file = match.group(1)
        full_path = asset_dir / stl_file
        
        if full_path.exists():
            xml_content = xml_content.replace(f'filename="{stl_file}"', f'filename="{full_path}"')
    
    return mujoco.MjModel.from_xml_string(xml_content)

print("="*80)
print("IK CONVERGENCE DIAGNOSIS")
print("="*80)

# Load model
model = load_model_with_assets()
data = mujoco.MjData(model)

# Load trajectories
base_traj = np.load("base_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

print("\nTrajectory Statistics:")
print(f"  Base trajectory range: x=[{base_traj[:,0].min():.3f}, {base_traj[:,0].max():.3f}]")
print(f"                         y=[{base_traj[:,1].min():.3f}, {base_traj[:,1].max():.3f}]")
print(f"                         z=[{base_traj[:,2].min():.3f}, {base_traj[:,2].max():.3f}]")
print(f"  Foot1 trajectory range: x=[{foot1_traj[:,0].min():.3f}, {foot1_traj[:,0].max():.3f}]")
print(f"                          y=[{foot1_traj[:,1].min():.3f}, {foot1_traj[:,1].max():.3f}]")
print(f"                          z=[{foot1_traj[:,2].min():.3f}, {foot1_traj[:,2].max():.3f}]")
print(f"  Foot2 trajectory range: x=[{foot2_traj[:,0].min():.3f}, {foot2_traj[:,0].max():.3f}]")
print(f"                          y=[{foot2_traj[:,1].min():.3f}, {foot2_traj[:,1].max():.3f}]")
print(f"                          z=[{foot2_traj[:,2].min():.3f}, {foot2_traj[:,2].max():.3f}]")

# Sample some poses and check FK
print("\nSampling robot poses with random joint angles...")
base_body_id = model.body(name='hip').id
foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id

# Check reachable workspace
print("\nScanning robot workspace with random poses...")
base_positions = []
foot1_positions = []
foot2_positions = []

np.random.seed(42)
for i in range(500):
    # Random joint angles for leg joints
    qpos = np.zeros(model.nq)
    qpos[0] = 0.0  # root_x
    qpos[1] = 0.0  # root_y
    qpos[2] = 0.40  # root_z (hip height)
    qpos[3:9] = np.random.uniform(-np.pi, np.pi, 6)  # leg angles
    
    # Forward kinematics
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    base_positions.append(data.xpos[base_body_id].copy())
    foot1_positions.append(data.site_xpos[foot1_site_id].copy())
    foot2_positions.append(data.site_xpos[foot2_site_id].copy())

base_positions = np.array(base_positions)
foot1_positions = np.array(foot1_positions)
foot2_positions = np.array(foot2_positions)

print(f"  Base reachable range: x=[{base_positions[:,0].min():.3f}, {base_positions[:,0].max():.3f}]")
print(f"                        y=[{base_positions[:,1].min():.3f}, {base_positions[:,1].max():.3f}]")
print(f"                        z=[{base_positions[:,2].min():.3f}, {base_positions[:,2].max():.3f}]")
print(f"  Foot1 reachable range: x=[{foot1_positions[:,0].min():.3f}, {foot1_positions[:,0].max():.3f}]")
print(f"                         y=[{foot1_positions[:,1].min():.3f}, {foot1_positions[:,1].max():.3f}]")
print(f"                         z=[{foot1_positions[:,2].min():.3f}, {foot1_positions[:,2].max():.3f}]")
print(f"  Foot2 reachable range: x=[{foot2_positions[:,0].min():.3f}, {foot2_positions[:,0].max():.3f}]")
print(f"                         y=[{foot2_positions[:,1].min():.3f}, {foot2_positions[:,1].max():.3f}]")
print(f"                         z=[{foot2_positions[:,2].min():.3f}, {foot2_positions[:,2].max():.3f}]")

# Check if trajectory targets fit in workspace
print("\nTrajectory vs Workspace Comparison:")
print("  Base:")
base_in_x = (base_traj[:,0].min() >= base_positions[:,0].min() and 
             base_traj[:,0].max() <= base_positions[:,0].max())
base_in_y = (base_traj[:,1].min() >= base_positions[:,1].min() and 
             base_traj[:,1].max() <= base_positions[:,1].max())
base_in_z = (base_traj[:,2].min() >= base_positions[:,2].min() and 
             base_traj[:,2].max() <= base_positions[:,2].max())
print(f"    X in range: {base_in_x}")
print(f"    Y in range: {base_in_y}")
print(f"    Z in range: {base_in_z}")

print("  Foot1:")
f1_in_x = (foot1_traj[:,0].min() >= foot1_positions[:,0].min() and 
           foot1_traj[:,0].max() <= foot1_positions[:,0].max())
f1_in_y = (foot1_traj[:,1].min() >= foot1_positions[:,1].min() and 
           foot1_traj[:,1].max() <= foot1_positions[:,1].max())
f1_in_z = (foot1_traj[:,2].min() >= foot1_positions[:,2].min() and 
           foot1_traj[:,2].max() <= foot1_positions[:,2].max())
print(f"    X in range: {f1_in_x}")
print(f"    Y in range: {f1_in_y}")
print(f"    Z in range: {f1_in_z}")

print("  Foot2:")
f2_in_x = (foot2_traj[:,0].min() >= foot2_positions[:,0].min() and 
           foot2_traj[:,0].max() <= foot2_positions[:,0].max())
f2_in_y = (foot2_traj[:,1].min() >= foot2_positions[:,1].min() and 
           foot2_traj[:,1].max() <= foot2_positions[:,1].max())
f2_in_z = (foot2_traj[:,2].min() >= foot2_positions[:,2].min() and 
           foot2_traj[:,2].max() <= foot2_positions[:,2].max())
print(f"    X in range: {f2_in_x}")
print(f"    Y in range: {f2_in_y}")
print(f"    Z in range: {f2_in_z}")

print("\n" + "="*80)

# Recommendation
if not (base_in_z and f1_in_z and f2_in_z):
    print("ISSUE: Some trajectory targets are outside workspace Z range!")
    print("       This is likely why IK isn't converging.")
    print("\nRECOMMENDATION: Adjust trajectory parameters (hip_height, stride_length, swing_clearance)")
elif not ((f1_in_x or f1_in_y) and (f2_in_x or f2_in_y)):
    print("ISSUE: Foot trajectories are outside reachable XY range!")
    print("\nRECOMMENDATION: Increase stride length or adjust gait parameters")
else:
    print("Workspace appears sufficient. IK convergence may be due to:")
    print("  - IK solver parameters (learning rate, damping) need tuning")
    print("  - Trajectory points may require different starting positions")
    print("  - Consider relaxing tolerance for partial IK convergence")
