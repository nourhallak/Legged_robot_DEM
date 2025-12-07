#!/usr/bin/env python3
"""
Simple Trajectory Playback
- Loads pre-computed walking trajectories
- Plays them back in MuJoCo viewer
"""
import numpy as np
import mujoco
from mujoco import viewer
import os
import re

def load_model_with_assets():
    """Load robot model with meshes"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # XML is in the same directory
    mjcf_path = os.path.join(script_dir, "legged_robot_ik.xml")
    
    with open(mjcf_path, 'r') as f:
        mjcf_content = f.read()
    
    # Meshes are in parent/Legged_robot/meshes
    main_dir = os.path.dirname(script_dir)
    meshes_dir = os.path.join(main_dir, "Legged_robot", "meshes")
    pattern = r'file="([^"]+\.STL)"'
    mesh_files = set(re.findall(pattern, mjcf_content))
    
    assets = {}
    for mesh_file in mesh_files:
        abs_path = os.path.join(meshes_dir, mesh_file)
        if os.path.exists(abs_path):
            with open(abs_path, 'rb') as f:
                assets[mesh_file] = f.read()
    
    model = mujoco.MjModel.from_xml_string(mjcf_content, assets=assets)
    return model

print("Loading model and trajectories...")
model = load_model_with_assets()
data = mujoco.MjData(model)

# Load trajectories from current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Try to load pre-computed joint angles first (best quality)
try:
    q_left = np.load(os.path.join(script_dir, 'q_left_feasible.npy'))
    q_right = np.load(os.path.join(script_dir, 'q_right_feasible.npy'))
    base = np.load(os.path.join(script_dir, 'base_feasible.npy'))
    print(f"✓ Loaded computed joint trajectories: {len(q_left)} steps")
    use_computed = True
except:
    print("✗ Joint trajectories not found in current folder")
    
    # Try parent folder (main DEM using Python)
    parent_dir = os.path.dirname(script_dir)
    try:
        q_left = np.load(os.path.join(parent_dir, 'q_left_feasible.npy'))
        q_right = np.load(os.path.join(parent_dir, 'q_right_feasible.npy'))
        base = np.load(os.path.join(parent_dir, 'base_feasible.npy'))
        print(f"✓ Loaded computed joint trajectories from parent: {len(q_left)} steps")
        use_computed = True
    except:
        print("✗ Joint trajectories not found in parent folder")
        print("✗ Using position-based trajectories (may vibrate)")
        use_computed = False

if use_computed:
    num_steps = len(q_left)
else:
    # Load position trajectories as fallback
    hip_traj = np.load(os.path.join(script_dir, 'hip_trajectory.npy'))
    foot1_traj = np.load(os.path.join(script_dir, 'foot1_trajectory.npy'))
    foot2_traj = np.load(os.path.join(script_dir, 'foot2_trajectory.npy'))
    num_steps = len(hip_traj)

print(f"Total steps: {num_steps}")
print(f"Starting walking visualization...")

# Set initial pose
if use_computed:
    data.qpos[0:3] = base[0]
    data.qpos[3:6] = q_left[0]
    data.qpos[6:9] = q_right[0]
else:
    data.qpos[0] = 0  # base x
    data.qpos[1] = hip_traj[0, 1]  # base y
    data.qpos[2] = 0  # base z rotation

mujoco.mj_forward(model, data)

# Play trajectory
with viewer.launch_passive(model, data) as v:
    step = 0
    while v.is_running():
        if use_computed:
            # Use pre-computed joint angles
            data.qpos[0:3] = base[step]
            data.qpos[3:6] = q_left[step]
            data.qpos[6:9] = q_right[step]
        else:
            # Use foot positions only
            data.qpos[0] = hip_traj[step, 0]
            data.qpos[1] = hip_traj[step, 1]
        
        mujoco.mj_forward(model, data)
        v.sync()
        
        step += 1
        if step >= num_steps:
            step = 0

print("Done!")
