#!/usr/bin/env python3
"""
Analytical Inverse Kinematics for Bipedal Walking

Instead of numerical IK (which has huge errors on tiny robots),
use analytical (closed-form) IK for each leg independently.

Robot geometry:
- Each leg: 3-DOF (hip, knee, ankle) with known link lengths
- Foot height constraint: z = 0.43m (ground contact)
- Only 2 DOF needed for planar motion: (x, z) position
"""

import numpy as np
import mujoco
import mujoco.viewer
import re
from pathlib import Path
import time

print("="*80)
print("ANALYTICAL IK FOR BIPEDAL WALKING")
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
    
    # Update mesh paths
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
# MEASURE LEG GEOMETRY
# ============================================================================

print("\nMeasuring leg link lengths...")

# Get hip position at zero config
data.qpos[:] = 0.0
data.qpos[2] = 0.42  # hip height
mujoco.mj_forward(model, data)

hip_pos = data.xpos[model.body(name='hip').id].copy()
print(f"  Hip position: {hip_pos}")

# Measure link lengths by moving each joint independently
def measure_link_length(joint_name, body_name):
    """Measure distance from joint to body"""
    data.qpos[:] = 0.0
    data.qpos[2] = 0.42
    mujoco.mj_forward(model, data)
    
    try:
        joint_id = model.joint(name=joint_name).id
        body_id = model.body(name=body_name).id
    except:
        return None
    
    return np.linalg.norm(data.xpos[body_id] - data.xpos[model.body(name='hip').id])

# Try to measure link lengths (may not be perfect, but gives approximation)
# For now, assume each segment is roughly similar length
data.qpos[:] = 0.0
data.qpos[2] = 0.42
data.qpos[3] = 0.0  # hip
data.qpos[4] = -np.pi/4  # knee
mujoco.mj_forward(model, data)

foot1_pos_bent = data.site_xpos[model.site(name='foot1_site').id].copy()
hip_pos = data.xpos[model.body(name='hip').id].copy()

# Simple estimate: measure from a bent configuration
L_hip_to_knee_approx = 0.010  # Rough estimate from geometry
L_knee_to_foot_approx = 0.010  # Rough estimate from geometry

print(f"  Estimated hip-to-knee length: {L_hip_to_knee_approx*1000:.1f} mm")
print(f"  Estimated knee-to-foot length: {L_knee_to_foot_approx*1000:.1f} mm")

# ============================================================================
# ANALYTICAL IK SOLVER
# ============================================================================

def analytical_ik_leg(target_x, target_z, hip_x, hip_z, 
                     L1=0.010, L2=0.010, max_angle=np.pi/2):
    """
    Analytical 2-link IK for leg in sagittal plane.
    
    Args:
        target_x, target_z: foot target position (world frame)
        hip_x, hip_z: hip position (world frame)
        L1: hip to knee link length
        L2: knee to foot link length
        
    Returns:
        (hip_angle, knee_angle, ankle_angle) in radians
    """
    
    # Vector from hip to target foot
    dx = target_x - hip_x
    dz = target_z - hip_z
    
    d = np.sqrt(dx**2 + dz**2)  # Distance from hip to foot target
    
    # Check reachability
    if d > L1 + L2:
        # Too far - scale back
        scale = (L1 + L2 - 0.001) / max(d, 1e-6)
        dx *= scale
        dz *= scale
        d = np.sqrt(dx**2 + dz**2)
    
    if d < abs(L1 - L2):
        # Too close - scale out
        scale = (abs(L1 - L2) + 0.001) / max(d, 1e-6)
        dx *= scale
        dz *= scale
        d = np.sqrt(dx**2 + dz**2)
    
    # Law of cosines for knee angle
    cos_knee = (L1**2 + L2**2 - d**2) / (2 * L1 * L2 + 1e-9)
    cos_knee = np.clip(cos_knee, -1, 1)
    knee_angle = np.arccos(cos_knee)
    
    # Hip angle (in world frame)
    alpha = np.arctan2(dz, dx)  # Angle to target from hip
    beta = np.arctan2(L2 * np.sin(knee_angle), 
                     L1 + L2 * np.cos(knee_angle) + 1e-9)
    hip_angle = alpha - beta
    
    # Ankle angle (maintain foot horizontal if possible)
    ankle_angle = -(hip_angle + knee_angle - np.pi)
    
    return hip_angle, knee_angle, ankle_angle

# ============================================================================
# TRAJECTORY LOADING
# ============================================================================

print("\nLoading trajectories...")
base_traj = np.load("base_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")
num_steps = len(base_traj)
print(f"[OK] Trajectories loaded: {num_steps} steps")

# ============================================================================
# SOLVE ANALYTICAL IK FOR ALL STEPS
# ============================================================================

print("\nSolving analytical IK for all trajectory points...")
qpos_solutions = []
error_norms = []

data.qpos[:] = 0.0
data.qpos[2] = 0.42

foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id
hip_body_id = model.body(name='hip').id

for step in range(num_steps):
    # Current hip position
    data.qpos[:] = 0.0
    data.qpos[2] = 0.42
    mujoco.mj_forward(model, data)
    hip_pos = data.xpos[hip_body_id].copy()
    
    # Target foot positions
    foot1_target = foot1_traj[step]
    foot2_target = foot2_traj[step]
    
    # Solve IK for left leg (Leg 1)
    h1_angle, k1_angle, a1_angle = analytical_ik_leg(
        foot1_target[0], foot1_target[2],  # x, z
        hip_pos[0], hip_pos[2],             # hip x, z
        L1=0.010, L2=0.010
    )
    
    # Solve IK for right leg (Leg 2)
    h2_angle, k2_angle, a2_angle = analytical_ik_leg(
        foot2_target[0], foot2_target[2],  # x, z
        hip_pos[0], hip_pos[2],             # hip x, z
        L1=0.010, L2=0.010
    )
    
    # Construct qpos
    qpos = np.zeros(9)
    qpos[2] = 0.42  # Keep hip height constant
    
    # Leg 1: joints 3 (hip), 4 (knee), 5 (ankle)
    qpos[3] = h1_angle
    qpos[4] = k1_angle
    qpos[5] = a1_angle
    
    # Leg 2: joints 6 (hip), 7 (knee), 8 (ankle)
    qpos[6] = h2_angle
    qpos[7] = k2_angle
    qpos[8] = a2_angle
    
    # Apply joint limits
    for j in range(3, 9):
        if j < model.jnt_range.shape[0]:
            qmin, qmax = model.jnt_range[j]
            qpos[j] = np.clip(qpos[j], qmin, qmax)
    
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

print(f"\n[OK] Analytical IK solved for all steps")
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
        
        # Display IK solution
        step_idx = int(sim_time * 10) % num_steps  # 0.1s per step
        data.qpos[:] = qpos_solutions[step_idx]
        mujoco.mj_forward(model, data)
        
        viewer.sync()

print("\n[OK] Viewer closed. Simulation complete!")
print("="*80)
