#!/usr/bin/env python3
"""
IK-based walking controller that tracks pre-planned trajectories (no viewer)
- Loads planned trajectories (hip, foot1, foot2)
- Uses IK solver to find joint angles that match trajectories
- Outputs tracking errors
"""
import numpy as np
import mujoco
import os
import re

# ============================================================================
# LOAD MODEL AND TRAJECTORIES
# ============================================================================

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

def get_sites(data, model):
    """Get current position of all sites and base"""
    # Hip is the base body (body 1)
    hip_id = model.body(name='hip').id
    hip_pos = data.xpos[hip_id].copy()
    
    # Feet are at foot sites
    foot1_id = model.site(name='foot1_site').id
    foot2_id = model.site(name='foot2_site').id
    
    foot1_pos = data.site_xpos[foot1_id].copy()
    foot2_pos = data.site_xpos[foot2_id].copy()
    
    return hip_pos, foot1_pos, foot2_pos

def compute_ik_solution(model, data, target_hip, target_foot1, target_foot2, 
                       max_iterations=50, tolerance=1e-4):
    """
    Solve IK to match target end-effector positions
    - Uses Jacobian-based numerical IK
    - Targets: hip (base), foot1, foot2
    """
    
    learning_rate = 0.1
    best_error = float('inf')
    best_qpos = data.qpos[6:12].copy()
    
    for iteration in range(max_iterations):
        # Current state
        hip_pos, foot1_pos, foot2_pos = get_sites(data, model)
        
        # Compute errors
        err_hip = target_hip - hip_pos
        err_foot1 = target_foot1 - foot1_pos
        err_foot2 = target_foot2 - foot2_pos
        
        total_error = np.linalg.norm(err_hip) + np.linalg.norm(err_foot1) + np.linalg.norm(err_foot2)
        
        if total_error < best_error:
            best_error = total_error
            best_qpos = data.qpos[6:12].copy()
        
        if total_error < tolerance:
            return data.qpos[6:12].copy(), total_error, iteration
        
        # Compute Jacobian (numerical differentiation)
        J = np.zeros((9, 6))  # 3 sites × 3 coords, 6 joints
        dq = 1e-6
        
        for j in range(6):
            data.qpos[6 + j] += dq
            mujoco.mj_kinematics(model, data)
            hip_pos_pert, foot1_pos_pert, foot2_pos_pert = get_sites(data, model)
            
            J[0:3, j] = (hip_pos_pert - hip_pos) / dq
            J[3:6, j] = (foot1_pos_pert - foot1_pos) / dq
            J[6:9, j] = (foot2_pos_pert - foot2_pos) / dq
            
            data.qpos[6 + j] -= dq
        
        # Damped least-squares (DLS) IK
        lambda_damp = 0.01
        JtJ = J.T @ J
        JtJ_damped = JtJ + lambda_damp * np.eye(6)
        
        try:
            dq_sol = np.linalg.solve(JtJ_damped, J.T @ np.concatenate([err_hip, err_foot1, err_foot2]))
        except np.linalg.LinAlgError:
            break
        
        # Update joints with bounds checking
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
# MAIN SIMULATION
# ============================================================================

print("Loading model...")
model = load_model_with_assets()
data = mujoco.MjData(model)

print("Loading trajectories...")
hip_traj = np.load('hip_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

num_steps = len(hip_traj)

print(f"\nStarting IK trajectory tracking ({num_steps} steps)...")
print("="*80)
print(f"{'Step':<6} {'Hip_X':<10} {'Hip_Z':<10} {'F1_Z':<10} {'F2_Z':<10} {'Error':<10} {'Iters':<6}")
print("="*80)

errors = []
iterations_per_step = []
joint_solutions = []

for step in range(num_steps):
    # Get target positions
    target_hip = hip_traj[step]
    target_foot1 = foot1_traj[step]
    target_foot2 = foot2_traj[step]
    
    # Solve IK
    joints, error, iters = compute_ik_solution(
        model, data,
        target_hip, target_foot1, target_foot2,
        max_iterations=50, tolerance=1e-4
    )
    
    # Apply joint solution
    data.qpos[6:12] = joints
    mujoco.mj_kinematics(model, data)
    
    # Get actual positions
    hip_pos, foot1_pos, foot2_pos = get_sites(data, model)
    
    errors.append(error)
    iterations_per_step.append(iters)
    joint_solutions.append(joints)
    
    # Print progress
    if step % 50 == 0 or step < 10:
        print(f"{step:<6} {hip_pos[0]:<10.5f} {hip_pos[2]:<10.5f} {foot1_pos[2]:<10.5f} {foot2_pos[2]:<10.5f} {error:<10.6f} {iters:<6}")

print("="*80)
print(f"\nSimulation Results:")
print(f"  Total steps: {num_steps}")
print(f"  Mean IK error: {np.mean(errors):.6f} m ({np.mean(errors)*1000:.2f} mm)")
print(f"  Max IK error: {np.max(errors):.6f} m ({np.max(errors)*1000:.2f} mm)")
print(f"  Min IK error: {np.min(errors):.6f} m ({np.min(errors)*1000:.2f} mm)")
print(f"  Errors > 10mm: {np.sum(np.array(errors) > 0.01)} steps")
print(f"  Errors > 50mm: {np.sum(np.array(errors) > 0.05)} steps")
print(f"  Mean iterations per step: {np.mean(iterations_per_step):.1f}")

# Save results
joint_solutions = np.array(joint_solutions)
np.save('joint_solutions_ik.npy', joint_solutions)
print(f"\nJoint solutions saved to: joint_solutions_ik.npy")
