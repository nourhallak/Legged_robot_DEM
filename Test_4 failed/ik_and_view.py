#!/usr/bin/env python3
"""
IK Solver + MuJoCo Viewer for Walking Trajectory

This script:
1. Loads the generated walking trajectories
2. Solves Inverse Kinematics for each trajectory point
3. Displays the robot walking in MuJoCo viewer
"""

import numpy as np
import mujoco
import mujoco.viewer
import re
from pathlib import Path
import time

print("="*80)
print("IK SOLVER + MUJOCO VIEWER FOR WALKING TRAJECTORIES")
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
# TRAJECTORY LOADING
# ============================================================================

print("Loading trajectories...")
base_traj = np.load("base_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")
num_steps = len(base_traj)
print(f"[OK] Trajectories loaded: {num_steps} steps")

# ============================================================================
# IK SOLVER
# ============================================================================

def compute_ik_solution(model, data, base_target, foot1_target, foot2_target, 
                       max_iterations=200, learning_rate=1.0, tolerance=5e-3):
    """
    Damped Least Squares IK Solver for bipedal walking
    
    Targets:
    - base_target: [x, y, z] position of hip body
    - foot1_target: [x, y, z] position of left foot
    - foot2_target: [x, y, z] position of right foot
    """
    
    try:
        foot1_site_id = model.site(name='foot1_site').id
        foot2_site_id = model.site(name='foot2_site').id
        base_body_id = model.body(name='hip').id
    except Exception as e:
        return data.qpos.copy(), False
    
    # Create a data object for computing Jacobian
    data_jac = mujoco.MjData(model)
    
    # Start from current qpos
    qpos = data.qpos.copy()
    
    epsilon = 1e-3
    damping = 0.01  # Very low damping for aggressive convergence
    
    for iteration in range(max_iterations):
        # Forward kinematics
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        
        # Current positions
        base_pos = data.xpos[base_body_id].copy()
        foot1_pos = data.site_xpos[foot1_site_id].copy()
        foot2_pos = data.site_xpos[foot2_site_id].copy()
        
        # Position errors - FEET + BASE HEIGHT (7 constraints for 6 DOF)
        # This overdetermined system helps stabilize the solution
        foot1_error = foot1_target - foot1_pos
        foot2_error = foot2_target - foot2_pos
        base_height_error = np.array([base_target[2] - base_pos[2]])  # Only Z constraint
        
        total_error = (np.linalg.norm(foot1_error) + 
                      np.linalg.norm(foot2_error) +
                      np.linalg.norm(base_height_error))
        
        if total_error < tolerance:
            return qpos, True
        
        # Compute Jacobian (numerical) - only for actuated joints
        act_joints = [3, 4, 5, 6, 7, 8]
        n_act_joints = len(act_joints)
        
        jacobian = np.zeros((7, n_act_joints))  # 7 constraints: feet (6) + base_height (1)
        
        for j_idx, j in enumerate(act_joints):
            # Perturb joint j
            qpos_plus = qpos.copy()
            qpos_plus[j] += epsilon
            
            data_jac.qpos[:] = qpos_plus
            mujoco.mj_forward(model, data_jac)
            
            foot1_pos_plus = data_jac.site_xpos[foot1_site_id]
            foot2_pos_plus = data_jac.site_xpos[foot2_site_id]
            base_pos_plus = data_jac.xpos[base_body_id]
            
            # Jacobian columns - feet (6) + base height (1)
            jacobian[0:3, j_idx] = (foot1_pos_plus - foot1_pos) / epsilon
            jacobian[3:6, j_idx] = (foot2_pos_plus - foot2_pos) / epsilon
            jacobian[6, j_idx] = (base_pos_plus[2] - base_pos[2]) / epsilon  # Base Z only
        
        # Damped Least Squares (overdetermined: 7x6)
        try:
            J_T = jacobian.T
            H = J_T @ jacobian + damping * np.eye(n_act_joints)
            H_inv = np.linalg.inv(H)
            jacobian_dls = H_inv @ J_T
            
            errors = np.concatenate([foot1_error, foot2_error, base_height_error])
            dq_act = learning_rate * jacobian_dls @ errors
            
        except np.linalg.LinAlgError:
            return qpos, False
        
        if not np.all(np.isfinite(dq_act)):
            return qpos, False
        
        # Update actuated joints
        for j_idx, j in enumerate(act_joints):
            qpos[j] += dq_act[j_idx]
        
        # Apply joint limits
        for j in act_joints:
            if j < model.jnt_range.shape[0]:
                qmin, qmax = model.jnt_range[j]
                qpos[j] = np.clip(qpos[j], qmin, qmax)
    
    return qpos, False

# ============================================================================
# SOLVE IK FOR ALL STEPS
# ============================================================================

print("\nSolving IK for all trajectory points...")
qpos_solutions = []
error_norms = []
success_count = 0

# Initialize to reasonable standing pose
data.qpos[:] = 0.0
data.qpos[2] = 0.42  # hip height (root_rz already at 0)

for step in range(num_steps):
    base_target = base_traj[step]
    foot1_target = foot1_traj[step]
    foot2_target = foot2_traj[step]
    
    # Solve IK
    qpos_solution, success = compute_ik_solution(
        model, data, 
        base_target, foot1_target, foot2_target,
        max_iterations=500, learning_rate=2.0, tolerance=1e-3
    )
    
    # Compute actual error
    data.qpos[:] = qpos_solution
    mujoco.mj_forward(model, data)
    
    base_pos = data.xpos[model.body(name='hip').id]
    foot1_pos = data.site_xpos[model.site(name='foot1_site').id]
    foot2_pos = data.site_xpos[model.site(name='foot2_site').id]
    
    base_err = np.linalg.norm(base_target - base_pos)
    foot1_err = np.linalg.norm(foot1_target - foot1_pos)
    foot2_err = np.linalg.norm(foot2_target - foot2_pos)
    total_err = base_err + foot1_err + foot2_err
    
    qpos_solutions.append(qpos_solution)
    error_norms.append(total_err)
    
    if success:
        success_count += 1
    
    # Use solution as warm start for next step
    data.qpos[:] = qpos_solution
    
    if (step + 1) % 50 == 0:
        avg_err = np.mean(error_norms[-50:])
        pct = 100 * success_count / (step + 1)
        print(f"  Processed {step+1:3d}/{num_steps} steps (success: {success_count:3d} = {pct:5.1f}%, avg_error: {avg_err:.6f})")

qpos_solutions = np.array(qpos_solutions)
avg_error = np.mean(error_norms)
max_error = np.max(error_norms)
print(f"\n[OK] IK solved for all steps")
print(f"     Average endpoint error: {avg_error:.4f}m ({avg_error*1000:.1f}mm)")
print(f"     Maximum endpoint error: {max_error:.4f}m ({max_error*1000:.1f}mm)")

# Save IK solutions for analysis
np.save("joint_solutions_ik.npy", qpos_solutions)
print(f"     Saved joint_solutions_ik.npy")

# ============================================================================
# MUJOCO VIEWER
# ============================================================================

print("\nStarting MuJoCo viewer...")
print("Controls:")
print("  - Click and drag to rotate view")
print("  - Right-click drag to pan")
print("  - Scroll to zoom")
print("  - Press 'Esc' or close window to exit")
print("-" * 80)

# Recreate model for viewer
model = load_model_with_assets()
data = mujoco.MjData(model)

# Set initial pose
data.qpos[:] = qpos_solutions[0]
mujoco.mj_forward(model, data)

# Parameters
dt = model.opt.timestep
sim_speed = 1.0  # Speed multiplier
frame_skip = 1   # Show every IK frame (reduced from 3 for smoother motion)

# Open viewer
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Configure view
        try:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        except:
            pass
        
        step_idx = 0
        frame_count = 0
        last_step = 0
        
        while viewer.is_running():
            # Get current IK target
            if step_idx < len(qpos_solutions):
                qpos_target = qpos_solutions[step_idx]
            else:
                # Loop
                step_idx = 0
                qpos_target = qpos_solutions[0]
            
            # Apply PD control on leg joints only
            Kp = 200.0   # Proportional gain (increased for stronger control)
            Kd = 20.0    # Derivative gain (increased for damping)
            
            # Control actuated joints (indices 3-8 corresponding to 6 motors)
            act_joints = [3, 4, 5, 6, 7, 8]
            for i, j in enumerate(act_joints):
                error = qpos_target[j] - data.qpos[j]
                vel = data.qvel[j]
                
                ctrl = Kp * error - Kd * vel
                data.ctrl[i] = np.clip(ctrl, -1.0, 1.0)
            
            # Step simulation
            mujoco.mj_step(model, data)
            frame_count += 1
            
            # Advance to next IK target periodically
            # Each IK trajectory step gets ~10 simulation frames to converge
            frames_per_step = 10
            if frame_count >= frames_per_step:
                frame_count = 0
                step_idx += 1
                if step_idx >= len(qpos_solutions):
                    step_idx = 0  # Loop back to beginning
            
            # Render
            viewer.sync()

except Exception as e:
    print(f"[ERROR] Viewer error: {e}")
    import traceback
    traceback.print_exc()

print("\n[OK] Viewer closed. Simulation complete!")
print("="*80)
