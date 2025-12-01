#!/usr/bin/env python3
"""
MPC-based Walking Controller for Legged Robot
Uses Model Predictive Control instead of trajectory-following IK
Optimizes joint commands to achieve walking goals naturally
"""
import numpy as np
import mujoco
import mujoco.viewer
import os
import re
from scipy.optimize import minimize
import time

def load_model_with_assets():
    """Load the robot model with all meshes"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.join(script_dir, "Legged_robot")
    meshes_dir = os.path.join(package_dir, "meshes")
    mjcf_output_path = os.path.join(script_dir, "legged_robot_ik.xml")

    if not os.path.exists(mjcf_output_path):
        raise FileNotFoundError(f"Model file not found at: {mjcf_output_path}")

    with open(mjcf_output_path, 'r', encoding='utf-8') as f:
        mjcf_content = f.read()

    MESH_PATTERN = r'file="([^"]+\.STL)"'
    all_mesh_files = set(re.findall(MESH_PATTERN, mjcf_content))

    assets = {}
    for mesh_file in all_mesh_files:
        abs_path = os.path.join(meshes_dir, mesh_file)
        if os.path.exists(abs_path):
            with open(abs_path, 'rb') as f:
                assets[mesh_file] = f.read()

    print("Loading MJCF model from string using explicit assets...")
    model = mujoco.MjModel.from_xml_string(mjcf_content, assets=assets)
    return model

def get_foot_positions(data, model):
    """Get current foot positions from sites"""
    foot1_site_id = model.site(name='foot1_site').id
    foot2_site_id = model.site(name='foot2_site').id
    
    foot1_pos = data.site_xpos[foot1_site_id].copy()
    foot2_pos = data.site_xpos[foot2_site_id].copy()
    
    return foot1_pos, foot2_pos

def mpc_controller(model, data, horizon=10, step_index=0):
    """
    MPC-based walking controller
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        horizon: Number of steps to predict ahead
        step_index: Current walking step
        
    Returns:
        joint_commands: Optimal joint angles for next control step
    """
    
    # Walking parameters
    cycle_length = 200  # Steps per full gait cycle
    phase = step_index % cycle_length
    
    # Target velocities and positions
    target_forward_vel = 0.05  # m/s forward speed
    target_height = 0.21  # Target foot contact height
    stance_foot_target_z = 0.210  # Ground contact
    swing_foot_target_z = 0.225  # Swing height (15mm clearance)
    
    # Current state
    base_pos = data.qpos[0:3].copy()
    base_quat = data.qpos[3:7].copy()
    joint_pos = data.qpos[6:12].copy()
    joint_vel = data.qvel[6:12].copy()
    
    foot1_pos, foot2_pos = get_foot_positions(data, model)
    
    # Determine which leg is in stance/swing phase
    is_foot1_stance = phase < 100
    
    def objective(joint_commands):
        """Objective function for MPC optimization"""
        # Create test data to simulate forward
        test_data = mujoco.MjData(model)
        test_data.qpos[:] = data.qpos.copy()
        test_data.qvel[:] = data.qvel.copy()
        
        # Apply joint commands (blend between current and target)
        alpha = 0.1  # Blending factor
        test_data.qpos[6:12] = joint_pos + alpha * (joint_commands - joint_pos)
        
        # Clamp to joint limits
        for i in range(6):
            test_data.qpos[6 + i] = np.clip(test_data.qpos[6 + i], 
                                            model.jnt_range[i, 0], 
                                            model.jnt_range[i, 1])
        
        # Forward kinematics
        mujoco.mj_forward(model, test_data)
        
        test_foot1, test_foot2 = get_foot_positions(test_data, model)
        
        # Objectives
        cost = 0.0
        
        # 1. Forward progress (encourage X movement)
        forward_cost = -target_forward_vel * (phase / cycle_length)  # Reference trajectory
        cost += 1.0 * (test_data.qpos[0] - forward_cost) ** 2
        
        # 2. Foot contact/clearance
        if is_foot1_stance:
            # Foot1 on ground (stance)
            stance_foot = test_foot1
            swing_foot = test_foot2
            target_stance_z = stance_foot_target_z
            target_swing_z = swing_foot_target_z
        else:
            # Foot2 on ground (stance)
            stance_foot = test_foot2
            swing_foot = test_foot1
            target_stance_z = stance_foot_target_z
            target_swing_z = swing_foot_target_z
        
        # Stance foot should be low (ground contact)
        stance_z_error = (stance_foot[2] - target_stance_z) ** 2
        cost += 5.0 * stance_z_error
        
        # Swing foot should be high (clearance)
        swing_z_error = max(0, target_swing_z - swing_foot[2]) ** 2  # Penalty if too low
        cost += 2.0 * swing_z_error
        
        # 3. Base height regulation
        target_base_z = 0.21  # Keep base above ground
        base_z_error = (test_data.qpos[2] - target_base_z) ** 2
        cost += 3.0 * base_z_error
        
        # 4. Joint smoothness (prefer small changes)
        smoothness_cost = np.sum((joint_commands - joint_pos) ** 2)
        cost += 0.1 * smoothness_cost
        
        return cost
    
    # Initial guess: current joint position
    x0 = joint_pos.copy()
    
    # Bounds for joint commands
    bounds = []
    for i in range(6):
        bounds.append((model.jnt_range[i, 0], model.jnt_range[i, 1]))
    
    # Optimize
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 50, 'ftol': 1e-4})
        joint_commands = result.x
    except:
        joint_commands = joint_pos.copy()
    
    return joint_commands

def main():
    print("Loading model...")
    model = load_model_with_assets()
    data = mujoco.MjData(model)
    
    # Initialize robot to standing position
    data.qpos[:] = 0
    data.qpos[2] = 0.21  # Base height
    data.qpos[6 + 1] = -0.8  # Hip joint
    data.qpos[6 + 2] = 0.8   # Knee joint
    data.qpos[6 + 4] = -0.8  # Right hip
    data.qpos[6 + 5] = 0.8   # Right knee
    
    mujoco.mj_forward(model, data)
    
    print("\n" + "="*80)
    print("MPC-BASED WALKING CONTROLLER")
    print("="*80)
    print("\nLaunching MuJoCo viewer...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera
        viewer.cam.azimuth = 0       # Face forward
        viewer.cam.elevation = -15   # Slight downward angle
        viewer.cam.distance = 0.4
        viewer.cam.lookat[:] = [0.01, 0.0, 0.22]
        
        mujoco.mj_resetData(model, data)
        
        step_count = 0
        max_steps = 400
        
        print(f"Starting MPC walking control for {max_steps} steps...")
        print(f"{'Step':<6} {'Base X':<10} {'Base Z':<10} {'Foot1 Z':<10} {'Foot2 Z':<10}")
        print("-" * 46)
        
        while viewer.is_running() and step_count < max_steps:
            # Compute MPC command
            joint_commands = mpc_controller(model, data, horizon=10, step_index=step_count)
            
            # Apply command with smoothing
            alpha = 0.05  # Low-pass filter for smooth motion
            for i in range(6):
                data.ctrl[i] = data.qpos[6 + i] + alpha * (joint_commands[i] - data.qpos[6 + i])
                data.ctrl[i] = np.clip(data.ctrl[i], model.jnt_range[i, 0], model.jnt_range[i, 1])
            
            # Simulate
            mujoco.mj_step(model, data)
            
            # Get foot positions
            foot1_site_id = model.site(name='foot1_site').id
            foot2_site_id = model.site(name='foot2_site').id
            foot1_z = data.site_xpos[foot1_site_id][2]
            foot2_z = data.site_xpos[foot2_site_id][2]
            
            # Print status every 50 steps
            if step_count % 50 == 0:
                print(f"{step_count:<6} {data.qpos[0]:<10.5f} {data.qpos[2]:<10.5f} {foot1_z:<10.5f} {foot2_z:<10.5f}")
            
            # Sync viewer
            viewer.sync()
            
            # Control timing
            time.sleep(0.01)
            
            step_count += 1
        
        print(f"\nSimulation complete! {step_count} steps executed.")
        print(f"Final base position: X={data.qpos[0]:.5f}m, Z={data.qpos[2]:.5f}m")
        
        # Keep viewer open
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()
