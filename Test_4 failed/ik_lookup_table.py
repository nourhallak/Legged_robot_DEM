#!/usr/bin/env python3
"""
Lookup Table IK for Bipedal Walking

Instead of analytical/numerical IK, build a lookup table (FK map) and 
use nearest-neighbor search for IK solving. Ultra-fast and accurate for small DOF.
"""

import numpy as np
import mujoco
import mujoco.viewer
import re
from pathlib import Path
import time
from scipy.spatial import cKDTree

print("="*80)
print("LOOKUP TABLE IK FOR BIPEDAL WALKING")
print("="*80)

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_with_assets(xml_path="legged_robot_ik.xml"):
    """Load MuJoCo model with correct asset paths"""
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

print("\nLoading robot model...")
model = load_model_with_assets()
data = mujoco.MjData(model)
print(f"[OK] Model loaded: {model.nq} DOF, {model.nu} actuators")

# ============================================================================
# BUILD LOOKUP TABLE (FK Map)
# ============================================================================

print("\nBuilding IK lookup table...")
print("  Sampling joint space for leg 1 and leg 2...")

# Joint ranges (from robot model)
j3_range = model.jnt_range[3]  # Leg1 hip
j4_range = model.jnt_range[4]  # Leg1 knee
j5_range = model.jnt_range[5]  # Leg1 ankle
j6_range = model.jnt_range[6]  # Leg2 hip
j7_range = model.jnt_range[7]  # Leg2 knee
j8_range = model.jnt_range[8]  # Leg2 ankle

print(f"  Joint 3 (hip1) range: [{j3_range[0]:.4f}, {j3_range[1]:.4f}]")
print(f"  Joint 4 (knee1) range: [{j4_range[0]:.4f}, {j4_range[1]:.4f}]")
print(f"  Joint 5 (ankle1) range: [{j5_range[0]:.4f}, {j5_range[1]:.4f}]")

# Sample joint space - ultra-fine grid for sub-mm accuracy
n_samples = 60  # 60^3 = 216000 configurations per leg
j3_samples = np.linspace(j3_range[0], j3_range[1], n_samples)
j4_samples = np.linspace(j4_range[0], j4_range[1], n_samples)
j5_samples = np.linspace(j5_range[0], j5_range[1], n_samples)

# For each leg separately
configs_leg1 = []  # List of (q3, q4, q5)
foot1_positions = []  # Corresponding foot1 positions

configs_leg2 = []
foot2_positions = []

foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id
hip_body_id = model.body(name='hip').id

# Build FK map for leg 1 - also vary hip height for ground contact
print("  Computing FK for Leg 1...")
z_samples = np.linspace(0.41, 0.44, 5)  # 5 hip height samples (fewer to keep speed reasonable)

for z_hip in z_samples:
    for q3 in j3_samples:
        for q4 in j4_samples:
            for q5 in j5_samples:
                # Set configuration
                data.qpos[:] = 0.0
                data.qpos[2] = z_hip  # Vary hip height
                data.qpos[3] = q3
                data.qpos[4] = q4
                data.qpos[5] = q5
                data.qpos[6] = 0  # Leg 2 neutral
                data.qpos[7] = 0
                data.qpos[8] = 0
                
                mujoco.mj_forward(model, data)
                
                foot_pos = data.site_xpos[foot1_site_id].copy()
                configs_leg1.append([q3, q4, q5, z_hip])  # Store hip height too
                foot1_positions.append(foot_pos)

configs_leg1 = np.array(configs_leg1)
foot1_positions = np.array(foot1_positions)

print(f"    Generated {len(configs_leg1)} configurations for Leg 1")
print(f"    Foot1 X range: [{foot1_positions[:, 0].min()*1000:.1f}, {foot1_positions[:, 0].max()*1000:.1f}] mm")
print(f"    Foot1 Z range: [{foot1_positions[:, 2].min()*1000:.1f}, {foot1_positions[:, 2].max()*1000:.1f}] mm")

# Build FK map for leg 2 - also vary hip height
print("  Computing FK for Leg 2...")
for z_hip in z_samples:
    for q6 in j3_samples:
        for q7 in j4_samples:
            for q8 in j5_samples:
                # Set configuration
                data.qpos[:] = 0.0
                data.qpos[2] = z_hip  # Vary hip height
                data.qpos[3] = 0  # Leg 1 neutral
                data.qpos[4] = 0
                data.qpos[5] = 0
                data.qpos[6] = q6
                data.qpos[7] = q7
                data.qpos[8] = q8
                
                mujoco.mj_forward(model, data)
                
                foot_pos = data.site_xpos[foot2_site_id].copy()
                configs_leg2.append([q6, q7, q8, z_hip])  # Store hip height too
                foot2_positions.append(foot_pos)

configs_leg2 = np.array(configs_leg2)
foot2_positions = np.array(foot2_positions)

print(f"    Generated {len(configs_leg2)} configurations for Leg 2")
print(f"    Foot2 X range: [{foot2_positions[:, 0].min()*1000:.1f}, {foot2_positions[:, 0].max()*1000:.1f}] mm")
print(f"    Foot2 Z range: [{foot2_positions[:, 2].min()*1000:.1f}, {foot2_positions[:, 2].max()*1000:.1f}] mm")

# Build KD-trees for fast nearest-neighbor search
print("  Building KD-trees for fast IK lookup...")
kdtree_leg1 = cKDTree(foot1_positions)
kdtree_leg2 = cKDTree(foot2_positions)
print("  [OK] KD-trees built")

# ============================================================================
# TRAJECTORY LOADING
# ============================================================================

print("\nLoading trajectories...")
base_traj = np.load("base_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")
num_steps = len(base_traj)
print(f"[OK] Trajectories loaded: {num_steps} steps")

print(f"\nTrajectory foot heights:")
print(f"  Foot1 Z range: [{foot1_traj[:,2].min()*1000:.1f}, {foot1_traj[:,2].max()*1000:.1f}] mm")
print(f"  Foot2 Z range: [{foot2_traj[:,2].min()*1000:.1f}, {foot2_traj[:,2].max()*1000:.1f}] mm")
print(f"Lookup table foot heights:")
print(f"  Foot1 Z range: [{foot1_positions[:,2].min()*1000:.1f}, {foot1_positions[:,2].max()*1000:.1f}] mm")
print(f"  Foot2 Z range: [{foot2_positions[:,2].min()*1000:.1f}, {foot2_positions[:,2].max()*1000:.1f}] mm")

# ============================================================================
# SOLVE IK USING LOOKUP TABLE
# ============================================================================

print("\nSolving IK using lookup table (nearest-neighbor)...")
qpos_solutions = []
error_norms = []

for step in range(num_steps):
    # Target foot positions
    foot1_target = foot1_traj[step]
    foot2_target = foot2_traj[step]
    
    # Query KD-trees
    dist1, idx1 = kdtree_leg1.query(foot1_target)
    dist2, idx2 = kdtree_leg2.query(foot2_target)
    
    # Get joint configurations
    config_leg1 = configs_leg1[idx1]
    config_leg2 = configs_leg2[idx2]
    
    # Construct full qpos
    qpos = np.zeros(9)
    # Use hip height from the better matching configuration
    qpos[2] = (config_leg1[3] + config_leg2[3]) / 2  # Average hip height
    qpos[3:6] = config_leg1[:3]
    qpos[6:9] = config_leg2[:3]
    
    # Verify solution
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    foot1_pos = data.site_xpos[foot1_site_id]
    foot2_pos = data.site_xpos[foot2_site_id]
    
    foot1_err = np.linalg.norm(foot1_target - foot1_pos)
    foot2_err = np.linalg.norm(foot2_target - foot2_pos)
    total_err = foot1_err + foot2_err
    
    qpos_solutions.append(qpos)
    error_norms.append(total_err)
    
    if (step + 1) % 50 == 0:
        print(f"  Processed {step+1:3d}/400 steps (avg_error: {np.mean(error_norms[-50:]):.6f})")

qpos_solutions = np.array(qpos_solutions)
error_norms = np.array(error_norms)

print(f"\n[OK] Lookup table IK solved for all steps")
print(f"     Average endpoint error: {error_norms.mean():.6f}m ({error_norms.mean()*1000:.2f}mm)")
print(f"     Maximum endpoint error: {error_norms.max():.6f}m ({error_norms.max()*1000:.2f}mm)")
print(f"     Saved joint_solutions_ik.npy")

np.save("joint_solutions_ik.npy", qpos_solutions)

# ============================================================================
# VISUALIZATION WITH MUJOCO VIEWER
# ============================================================================

print("\nStarting MuJoCo viewer...")
print("Controls:")
print("  - Click and drag to rotate view")
print("  - Right-click drag to pan")
print("  - Scroll to zoom")
print("  - Press 'Esc' or close window to exit")
print("-" * 80)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    step_idx = 0
    
    while viewer.is_running():
        sim_time = time.time() - start_time
        
        # Display IK solution (slow playback)
        step_idx = int(sim_time * 2) % num_steps  # 0.5s per step
        data.qpos[:] = qpos_solutions[step_idx]
        mujoco.mj_forward(model, data)
        
        viewer.sync()

print("\n[OK] Viewer closed. Simulation complete!")
print("="*80)
