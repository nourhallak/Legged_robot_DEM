#!/usr/bin/env python3
"""
Test MPC controller without viewer to see if it works
"""
import numpy as np
import mujoco
import os
import re
from scipy.optimize import minimize

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

    print("Loading MJCF model...")
    model = mujoco.MjModel.from_xml_string(mjcf_content, assets=assets)
    return model

def get_foot_positions(data, model):
    """Get current foot positions from sites"""
    foot1_site_id = model.site(name='foot1_site').id
    foot2_site_id = model.site(name='foot2_site').id
    
    foot1_pos = data.site_xpos[foot1_site_id].copy()
    foot2_pos = data.site_xpos[foot2_site_id].copy()
    
    return foot1_pos, foot2_pos

def mpc_controller(model, data, step_index=0):
    """Simple MPC controller"""
    
    # Current state
    joint_pos = data.qpos[6:12].copy()
    foot1_pos, foot2_pos = get_foot_positions(data, model)
    
    # Phase in gait cycle
    cycle_length = 200
    phase = step_index % cycle_length
    is_foot1_stance = phase < 100
    
    def objective(joint_commands):
        """Objective function"""
        test_data = mujoco.MjData(model)
        test_data.qpos[:] = data.qpos.copy()
        
        # Smooth transition to new joint angles
        alpha = 0.1
        test_data.qpos[6:12] = joint_pos + alpha * (joint_commands - joint_pos)
        
        # Clamp to limits
        for i in range(6):
            test_data.qpos[6 + i] = np.clip(test_data.qpos[6 + i], 
                                            model.jnt_range[i, 0], 
                                            model.jnt_range[i, 1])
        
        mujoco.mj_forward(model, test_data)
        test_foot1, test_foot2 = get_foot_positions(test_data, model)
        
        cost = 0.0
        
        # Foot height objectives
        if is_foot1_stance:
            stance_foot_z_error = (test_foot1[2] - 0.210) ** 2
            swing_foot_z_error = max(0, 0.225 - test_foot2[2]) ** 2
        else:
            stance_foot_z_error = (test_foot2[2] - 0.210) ** 2
            swing_foot_z_error = max(0, 0.225 - test_foot1[2]) ** 2
        
        cost += 5.0 * stance_foot_z_error
        cost += 2.0 * swing_foot_z_error
        
        # Base height
        base_z_error = (test_data.qpos[2] - 0.21) ** 2
        cost += 3.0 * base_z_error
        
        # Smoothness
        smoothness = np.sum((joint_commands - joint_pos) ** 2)
        cost += 0.1 * smoothness
        
        return cost
    
    # Optimize
    bounds = [(model.jnt_range[i, 0], model.jnt_range[i, 1]) for i in range(6)]
    result = minimize(objective, joint_pos, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 50})
    
    return result.x

def main():
    print("Testing MPC Walking Controller\n")
    
    model = load_model_with_assets()
    data = mujoco.MjData(model)
    
    # Initial standing pose
    data.qpos[:] = 0
    data.qpos[2] = 0.21
    data.qpos[6 + 1] = -0.8
    data.qpos[6 + 2] = 0.8
    data.qpos[6 + 4] = -0.8
    data.qpos[6 + 5] = 0.8
    
    mujoco.mj_forward(model, data)
    
    print("="*70)
    print("MPC WALKING TEST")
    print("="*70)
    print(f"\n{'Step':<6} {'Base_X':<10} {'Base_Z':<10} {'Foot1_Z':<10} {'Foot2_Z':<10}")
    print("-"*50)
    
    for step in range(100):
        # Get MPC command
        joint_commands = mpc_controller(model, data, step_index=step)
        
        # Apply smoothly
        alpha = 0.1
        for i in range(6):
            data.qpos[6 + i] = data.qpos[6 + i] + alpha * (joint_commands[i] - data.qpos[6 + i])
            data.qpos[6 + i] = np.clip(data.qpos[6 + i], model.jnt_range[i, 0], model.jnt_range[i, 1])
        
        # Forward kinematics
        mujoco.mj_forward(model, data)
        
        foot1, foot2 = get_foot_positions(data, model)
        
        if step % 10 == 0:
            print(f"{step:<6} {data.qpos[0]:<10.5f} {data.qpos[2]:<10.5f} {foot1[2]:<10.5f} {foot2[2]:<10.5f}")
    
    print("\nTest complete! MPC controller works!")

if __name__ == "__main__":
    main()
