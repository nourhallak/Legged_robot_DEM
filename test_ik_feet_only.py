#!/usr/bin/env python3
"""
Correct IK-based trajectory tracker
- Directly sets hip position (it's the floating base, not solved by IK)
- Uses IK to solve for joint angles to reach target foot positions
- This is the correct approach for floating-base robots
"""
import numpy as np
import mujoco
import os
import re

def load_model_with_assets():
    """Load robot model with mesh assets"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mjcf_path = os.path.join(script_dir, "legged_robot_ik.xml")
    
    with open(mjcf_path, 'r') as f:
        mjcf_content = f.read()
    
    meshes_dir = os.path.join(script_dir, "Legged_robot", "meshes")
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

def get_feet_positions(data, model):
    """Get current foot positions"""
    foot1_id = model.site(name='foot1_site').id
    foot2_id = model.site(name='foot2_site').id
    
    foot1_pos = data.site_xpos[foot1_id].copy()
    foot2_pos = data.site_xpos[foot2_id].copy()
    
    return foot1_pos, foot2_pos

def compute_ik_feet_only(model, data, target_foot1, target_foot2, 
                         max_iterations=50, tolerance=1e-4):
    """
    Solve IK for foot positions only (6 DOF for 2 feet = 6 constraints)
    - Fixes the hip position (set via qpos[0:3])
    - Solves 6 joint angles to reach target foot positions
    """
    
    learning_rate = 0.1
    best_error = float('inf')
    best_qpos = data.qpos[6:12].copy()
    
    for iteration in range(max_iterations):
        # Current foot positions
        foot1_pos, foot2_pos = get_feet_positions(data, model)
        
        # Compute errors
        err_foot1 = target_foot1 - foot1_pos
        err_foot2 = target_foot2 - foot2_pos
        
        total_error = np.linalg.norm(err_foot1) + np.linalg.norm(err_foot2)
        
        if total_error < best_error:
            best_error = total_error
            best_qpos = data.qpos[6:12].copy()
        
        if total_error < tolerance:
            return data.qpos[6:12].copy(), total_error, iteration
        
        # Compute Jacobian for feet only (6x6)
        J = np.zeros((6, 6))
        dq = 1e-6
        
        for j in range(6):
            data.qpos[6 + j] += dq
            mujoco.mj_kinematics(model, data)
            foot1_pert, foot2_pert = get_feet_positions(data, model)
            
            J[0:3, j] = (foot1_pert - foot1_pos) / dq
            J[3:6, j] = (foot2_pert - foot2_pos) / dq
            
            data.qpos[6 + j] -= dq
        
        # Damped least-squares IK
        lambda_damp = 0.01
        JtJ = J.T @ J
        JtJ_damped = JtJ + lambda_damp * np.eye(6)
        
        try:
            dq_sol = np.linalg.solve(JtJ_damped, J.T @ np.concatenate([err_foot1, err_foot2]))
        except np.linalg.LinAlgError:
            break
        
        # Update joints with bounds
        joint_limits = np.array([
            [-1.57, 1.57],   # hip_link_2_1
            [-2.0944, 1.0472],  # link_2_1_link_1_1
            [-1.57, 1.57],   # link_1_1_foot_1
            [-1.57, 1.57],   # hip_link_2_2
            [-2.0944, 1.0472],  # link_2_2_link_1_2
            [-1.57, 1.57],   # link_1_2_foot_2
        ])
        
        new_joints = data.qpos[6:12] + learning_rate * dq_sol
        new_joints = np.clip(new_joints, joint_limits[:, 0], joint_limits[:, 1])
        
        data.qpos[6:12] = new_joints
        mujoco.mj_kinematics(model, data)
    
    return best_qpos, best_error, max_iterations

# ============================================================================
# MAIN
# ============================================================================

print("Loading model...")
model = load_model_with_assets()
data = mujoco.MjData(model)

print("Loading trajectories...")
hip_traj = np.load('hip_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

num_steps = len(hip_traj)

print(f"\nStarting trajectory tracking ({num_steps} steps)...")
print("=" * 80)
print(f"{'Step':<6} {'Hip_Z':<10} {'F1_Z_T':<10} {'F1_Z_A':<10} {'F2_Z_T':<10} {'F2_Z_A':<10} {'Error':<10}")
print("=" * 80)

errors = []
iterations_per_step = []

for step in range(num_steps):
    # Set hip position directly (floating base)
    target_hip = hip_traj[step]
    data.qpos[0:3] = target_hip
    data.qpos[3:7] = [1, 0, 0, 0]  # Identity quaternion
    
    # Get target foot positions
    target_foot1 = foot1_traj[step]
    target_foot2 = foot2_traj[step]
    
    # Solve IK for feet
    joints, error, iters = compute_ik_feet_only(
        model, data,
        target_foot1, target_foot2,
        max_iterations=50, tolerance=1e-5
    )
    
    # Apply joint solution
    data.qpos[6:12] = joints
    mujoco.mj_kinematics(model, data)
    
    # Get actual positions
    foot1_act, foot2_act = get_feet_positions(data, model)
    
    errors.append(error)
    iterations_per_step.append(iters)
    
    # Print progress
    if step % 50 == 0 or step < 10:
        print(f"{step:<6} {target_hip[2]:<10.5f} {target_foot1[2]:<10.5f} {foot1_act[2]:<10.5f} {target_foot2[2]:<10.5f} {foot2_act[2]:<10.5f} {error:<10.6f}")

print("=" * 80)
print(f"\nResults:")
print(f"  Total steps: {num_steps}")
print(f"  Mean IK error: {np.mean(errors):.6f} m ({np.mean(errors)*1000:.2f} mm)")
print(f"  Max IK error: {np.max(errors):.6f} m ({np.max(errors)*1000:.2f} mm)")
print(f"  Min IK error: {np.min(errors):.6f} m ({np.min(errors)*1000:.2f} mm)")
print(f"  Errors > 5mm: {np.sum(np.array(errors) > 0.005)}")
print(f"  Errors > 10mm: {np.sum(np.array(errors) > 0.010)}")
print(f"  Mean iterations: {np.mean(iterations_per_step):.1f}")
