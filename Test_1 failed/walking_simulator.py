#!/usr/bin/env python3
"""
Complete walking simulation with trajectory planning and IK tracking
- Generates smooth walking trajectories for hip and feet
- Uses IK to solve joint angles to track trajectories
- Displays walking motion
"""
import numpy as np
import mujoco
import os
import re

def load_model_with_assets():
    """Load robot model"""
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
    """Get foot positions"""
    foot1_id = model.site(name='foot1_site').id
    foot2_id = model.site(name='foot2_site').id
    return data.site_xpos[foot1_id].copy(), data.site_xpos[foot2_id].copy()

def compute_ik(model, data, target_foot1, target_foot2, max_iterations=50):
    """Solve IK for feet"""
    learning_rate = 0.1
    best_error = float('inf')
    best_qpos = data.qpos[6:12].copy()
    
    for _ in range(max_iterations):
        foot1_pos, foot2_pos = get_feet_positions(data, model)
        err_foot1 = target_foot1 - foot1_pos
        err_foot2 = target_foot2 - foot2_pos
        total_error = np.linalg.norm(err_foot1) + np.linalg.norm(err_foot2)
        
        if total_error < best_error:
            best_error = total_error
            best_qpos = data.qpos[6:12].copy()
        
        if total_error < 1e-5:
            return data.qpos[6:12].copy(), total_error
        
        # Jacobian
        J = np.zeros((6, 6))
        dq = 1e-6
        for j in range(6):
            data.qpos[6 + j] += dq
            mujoco.mj_kinematics(model, data)
            f1p, f2p = get_feet_positions(data, model)
            J[0:3, j] = (f1p - foot1_pos) / dq
            J[3:6, j] = (f2p - foot2_pos) / dq
            data.qpos[6 + j] -= dq
        
        lambda_damp = 0.01
        JtJ_damped = J.T @ J + lambda_damp * np.eye(6)
        try:
            dq_sol = np.linalg.solve(JtJ_damped, J.T @ np.concatenate([err_foot1, err_foot2]))
        except:
            break
        
        joint_limits = np.array([[-1.57, 1.57], [-2.0944, 1.0472], [-1.57, 1.57],
                                 [-1.57, 1.57], [-2.0944, 1.0472], [-1.57, 1.57]])
        new_joints = data.qpos[6:12] + learning_rate * dq_sol
        new_joints = np.clip(new_joints, joint_limits[:, 0], joint_limits[:, 1])
        data.qpos[6:12] = new_joints
        mujoco.mj_kinematics(model, data)
    
    return best_qpos, best_error

# ============================================================================
# MAIN SIMULATION
# ============================================================================

print("Initializing walking simulation...")
model = load_model_with_assets()
data = mujoco.MjData(model)

# Load or generate trajectories
if not os.path.exists('hip_trajectory.npy'):
    print("ERROR: Trajectories not found. Run trajectory_planner.py first.")
    exit(1)

hip_traj = np.load('hip_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

num_steps = len(hip_traj)

# Run simulation
print(f"\nRunning {num_steps}-step walking simulation...")
print("="*70)

errors = []
total_error_distance = 0

for step in range(num_steps):
    # Set base position (directly, no IK needed)
    target_hip = hip_traj[step]
    data.qpos[0:3] = target_hip
    data.qpos[3:7] = [1, 0, 0, 0]
    
    # Solve IK for feet
    target_foot1 = foot1_traj[step]
    target_foot2 = foot2_traj[step]
    joints, error = compute_ik(model, data, target_foot1, target_foot2, max_iterations=50)
    
    # Apply solution
    data.qpos[6:12] = joints
    mujoco.mj_kinematics(model, data)
    mujoco.mj_step(model, data)
    
    errors.append(error)
    total_error_distance += error
    
    if step % 100 == 0:
        foot1_act, foot2_act = get_feet_positions(data, model)
        print(f"Step {step:3d}: Hip_Z={target_hip[2]:.4f}m, F1_Z={foot1_act[2]:.4f}m (target {target_foot1[2]:.4f}m), Error={error:.4f}m")

print("="*70)
print("\n[OK] Walking Simulation Complete!")
print(f"  Total steps: {num_steps}")
print(f"  Total X distance: {hip_traj[-1, 0]:.4f}m")
print(f"  Mean IK error: {np.mean(errors):.6f}m ({np.mean(errors)*1000:.2f}mm)")
print(f"  Max IK error: {np.max(errors):.6f}m ({np.max(errors)*1000:.2f}mm)")
print(f"  Total steps with error < 10mm: {np.sum(np.array(errors) < 0.01)}/{num_steps}")
print()
print("Trajectories:")
print(f"  hip_trajectory.npy - Base position for each step")
print(f"  foot1_trajectory.npy - Left foot position for each step")
print(f"  foot2_trajectory.npy - Right foot position for each step")
print()
print("This completes the trajectory planning pipeline!")
