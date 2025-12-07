#!/usr/bin/env python3
"""
Walking Motion Visualizer
- Loads pre-computed trajectories
- Solves IK at each step
- Shows the robot walking with real-time visualization
"""
import numpy as np
import mujoco
import mujoco.viewer
import os
import re

def load_model_with_assets():
    """Load robot model with meshes"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # XML is in the same directory (Test_1 failed)
    mjcf_path = os.path.join(script_dir, "legged_robot_ik.xml")
    print(f"Script dir: {script_dir}")
    print(f"Looking for XML at: {mjcf_path}")
    print(f"File exists: {os.path.exists(mjcf_path)}")
    
    with open(mjcf_path, 'r') as f:
        mjcf_content = f.read()
    
    # Meshes are in parent/Legged_robot/meshes
    main_dir = os.path.dirname(script_dir)
    meshes_dir = os.path.join(main_dir, "Legged_robot", "meshes")
    print(f"Looking for meshes at: {meshes_dir}")
    pattern = r'file="([^"]+\.STL)"'
    mesh_files = set(re.findall(pattern, mjcf_content))
    
    assets = {}
    for mesh_file in mesh_files:
        abs_path = os.path.join(meshes_dir, mesh_file)
        if os.path.exists(abs_path):
            with open(abs_path, 'rb') as f:
                assets[mesh_file] = f.read()
                print(f"Loaded: {mesh_file}")
    
    print(f"Loaded {len(assets)} mesh assets")
    model = mujoco.MjModel.from_xml_string(mjcf_content, assets=assets)
    return model

def get_feet_positions(data, model):
    """Get foot site positions"""
    foot1_id = model.site(name='foot1_site').id
    foot2_id = model.site(name='foot2_site').id
    return data.site_xpos[foot1_id].copy(), data.site_xpos[foot2_id].copy()

def compute_ik(model, data, target_foot1, target_foot2, max_iterations=50):
    """Solve IK for feet positions"""
    learning_rate = 0.15
    best_error = float('inf')
    best_qpos = data.qpos[6:12].copy()
    
    for iteration in range(max_iterations):
        foot1_pos, foot2_pos = get_feet_positions(data, model)
        err_foot1 = target_foot1 - foot1_pos
        err_foot2 = target_foot2 - foot2_pos
        total_error = np.linalg.norm(err_foot1) + np.linalg.norm(err_foot2)
        
        if total_error < best_error:
            best_error = total_error
            best_qpos = data.qpos[6:12].copy()
        
        if total_error < 5e-5:  # Stricter convergence
            return data.qpos[6:12].copy(), total_error
        
        # Compute Jacobian
        J = np.zeros((6, 6))
        dq = 1e-6
        for j in range(6):
            data.qpos[6 + j] += dq
            mujoco.mj_kinematics(model, data)
            f1p, f2p = get_feet_positions(data, model)
            J[0:3, j] = (f1p - foot1_pos) / dq
            J[3:6, j] = (f2p - foot2_pos) / dq
            data.qpos[6 + j] -= dq
        
        # Damped least-squares with adaptive damping
        lambda_damp = 0.001 if iteration < 10 else 0.01  # Less damping initially
        JtJ_damped = J.T @ J + lambda_damp * np.eye(6)
        try:
            dq_sol = np.linalg.solve(JtJ_damped, J.T @ np.concatenate([err_foot1, err_foot2]))
        except:
            break
        
        # Update with joint limits
        joint_limits = np.array([[-1.57, 1.57], [-2.0944, 1.0472], [-1.57, 1.57],
                                 [-1.57, 1.57], [-2.0944, 1.0472], [-1.57, 1.57]])
        new_joints = data.qpos[6:12] + learning_rate * dq_sol
        new_joints = np.clip(new_joints, joint_limits[:, 0], joint_limits[:, 1])
        
        data.qpos[6:12] = new_joints
        mujoco.mj_kinematics(model, data)
    
    return best_qpos, best_error

# ============================================================================
# MAIN VISUALIZATION
# ============================================================================

print("Loading model and trajectories...")
model = load_model_with_assets()
data = mujoco.MjData(model)

# Load trajectories
hip_traj = np.load('hip_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

num_steps = len(hip_traj)

print(f"Starting walking visualization ({num_steps} steps)...")
print("Controls: Space to play/pause, arrow keys to step, ESC to quit")
print()

# Create viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set camera
    viewer.cam.azimuth = 0
    viewer.cam.elevation = -20
    viewer.cam.lookat = np.array([0.005, 0.0, 0.22])
    viewer.cam.distance = 0.5
    
    # Simulation loop
    step_idx = 0
    paused = False
    
    while viewer.is_running():
        # Get current trajectory targets
        target_hip = hip_traj[step_idx]
        target_foot1 = foot1_traj[step_idx]
        target_foot2 = foot2_traj[step_idx]
        
        # Set hip position (floating base)
        data.qpos[0:3] = target_hip
        data.qpos[3:7] = [1, 0, 0, 0]  # Identity quaternion
        
        # Solve IK for feet with better convergence
        joints, error = compute_ik(model, data, target_foot1, target_foot2, max_iterations=50)
        data.qpos[6:12] = joints
        
        # Update kinematics
        mujoco.mj_kinematics(model, data)
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Print info every 50 steps
        if step_idx % 50 == 0:
            f1_pos, f2_pos = get_feet_positions(data, model)
            print(f"Step {step_idx:3d}: Hip_Z={target_hip[2]:.4f}m, "
                  f"Foot1_Z={f1_pos[2]:.4f}m, Foot2_Z={f2_pos[2]:.4f}m, "
                  f"IK_Error={error*1000:.1f}mm")
        
        # Auto-advance step
        if not paused:
            step_idx = (step_idx + 1) % num_steps

print("\nWalking visualization complete!")
