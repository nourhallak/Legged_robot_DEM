import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import re

# --- Utility Functions for Asset Loading ---

def load_model_with_assets():
    """
    Loads the converted MJCF model ('legged_robot_ik.xml') using the 
    asset loading method to ensure meshes and sites are correctly included.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.join(script_dir, "Legged_robot")
    meshes_dir = os.path.join(package_dir, "meshes")
    mjcf_output_path = os.path.join(script_dir, "legged_robot_ik.xml")

    # 1. Check if the converted MJCF exists
    if not os.path.exists(mjcf_output_path):
        raise FileNotFoundError(
            f"Converted model file not found at: {mjcf_output_path}. "
            "Please run 'conversion.py' first to generate it."
        )

    # 2. Read the converted MJCF content
    with open(mjcf_output_path, 'r', encoding='utf-8') as f:
        mjcf_content = f.read()

    # 3. Find all required mesh filenames in the MJCF content
    MESH_PATTERN = r'file="([^"]+\.STL)"'
    all_mesh_files = set(re.findall(MESH_PATTERN, mjcf_content))

    if not all_mesh_files:
        print("Warning: No mesh files found referenced in the converted MJCF.")

    # 4. Build the Assets Dictionary
    assets = {}
    for mesh_file in all_mesh_files:
        abs_path = os.path.join(meshes_dir, mesh_file)
        
        if not os.path.exists(abs_path):
            print(f"Warning: Mesh file '{mesh_file}' referenced in MJCF not found on disk at {abs_path}")
            continue

        with open(abs_path, 'rb') as f:
            assets[mesh_file] = f.read()

    # 5. Load the model from string, passing the assets dictionary
    print("Loading MJCF model from string using explicit assets...")
    model = mujoco.MjModel.from_xml_string(mjcf_content, assets=assets)
    return model

# --- Inverse Kinematics using Numerical Jacobian ---

def compute_ik_solution(model, data, base_target_pos, com_target_pos, foot1_target_pos, foot2_target_pos, max_iterations=20, tolerance=0.005):
    """
    Solves inverse kinematics using numerical Jacobian and gradient descent.
    Tracks base + feet end-effectors only (9 DOF = 9 constraints).
    COM constraint removed as it was causing overconstrained system.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        base_target_pos: Target position for floating base [3]
        com_target_pos: IGNORED (kept for API compatibility)
        foot1_target_pos: Target position for foot 1 [3]
        foot2_target_pos: Target position for foot 2 [3]
        max_iterations: Maximum iterations for IK solver
        tolerance: Position error tolerance for convergence
        
    Returns:
        qpos_solution: Solved joint positions
        success: Boolean indicating if IK converged
    """
    
    # Get site IDs
    try:
        foot1_site_id = model.site(name='foot1_site').id
        foot2_site_id = model.site(name='foot2_site').id
    except Exception as e:
        raise ValueError(f"Could not find required sites: {e}")
    
    # Store initial qpos
    qpos_init = data.qpos.copy()
    qpos = qpos_init.copy()
    
    # Optimization parameters
    alpha = 0.08  # Learning rate
    epsilon = 1e-6  # Perturbation for numerical Jacobian
    
    for iteration in range(max_iterations):
        # Set current qpos and compute forward kinematics
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        
        # Get current positions
        base_pos = qpos[0:3].copy()  # Current base position
        foot1_pos = data.site_xpos[foot1_site_id].copy()
        foot2_pos = data.site_xpos[foot2_site_id].copy()
        
        # Compute position errors - base + feet only (COM removed to avoid overconstrained system)
        base_error = base_target_pos - base_pos
        foot1_error = foot1_target_pos - foot1_pos
        foot2_error = foot2_target_pos - foot2_pos
        
        # Overall error magnitude
        total_error = (np.linalg.norm(base_error) + 
                      np.linalg.norm(foot1_error) + np.linalg.norm(foot2_error))
        
        if total_error < tolerance:
            return qpos, True
        
        # Compute numerical Jacobian for: [base_x, base_y, base_z, joint0-5]
        # 9 outputs: base(3) + foot1(3) + foot2(3)
        # 9 variables: base_xyz(3) + actuated_joints(6)
        jacobian = np.zeros((9, 9))
        
        # Jacobian for base: identity for base XYZ positions
        jacobian[0:3, 0:3] = np.eye(3)
        
        # For each actuated joint, compute effect on end-effectors
        for j in range(6):  # Only actuated joints
            qpos_plus = qpos.copy()
            qpos_plus[6 + j] += epsilon  # Perturb actuated joint j
            
            data.qpos[:] = qpos_plus
            mujoco.mj_forward(model, data)
            
            pos_plus_foot1 = data.site_xpos[foot1_site_id].copy()
            pos_plus_foot2 = data.site_xpos[foot2_site_id].copy()
            
            # Finite difference - store in columns 3-8 (for actuated joints)
            jacobian[3:6, 3 + j] = (pos_plus_foot1 - foot1_pos) / epsilon
            jacobian[6:9, 3 + j] = (pos_plus_foot2 - foot2_pos) / epsilon
        
        # Compute pseudo-inverse of Jacobian
        try:
            jacobian_pinv = np.linalg.pinv(jacobian, rcond=1e-6)
        except:
            return qpos, False
        
        # Stack errors - base + feet only
        errors = np.concatenate([base_error, foot1_error, foot2_error])
        
        # Compute update: base_xyz + actuated_joints
        dq_all = alpha * jacobian_pinv @ errors
        
        # Check for NaN or Inf
        if np.any(~np.isfinite(dq_all)):
            return qpos, False
        
        # Clip to prevent unstable jumps
        dq_all = np.clip(dq_all, -0.2, 0.2)
        
        # Update base XYZ
        qpos[0:3] += dq_all[0:3]
        
        # Update actuated joints
        for i in range(6):
            qpos[6 + i] = qpos[6 + i] + dq_all[3 + i]
        
        # Clamp actuated joints to their limits
        for i in range(6):
            joint_idx = i
            if joint_idx < model.jnt_range.shape[0]:
                qpos[6 + i] = np.clip(qpos[6 + i], model.jnt_range[joint_idx, 0], model.jnt_range[joint_idx, 1])
        
        # Keep base orientation fixed (quat = [1, 0, 0, 0])
        qpos[3:7] = [1, 0, 0, 0]
    
    return qpos, False

def main():
    # --- Load Model and Trajectories ---
    
    print("Loading trajectories...")
    try:
        # Load pre-calculated trajectory files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_traj = np.load(os.path.join(script_dir, "base_trajectory.npy"))
        com_traj = np.load(os.path.join(script_dir, "com_trajectory.npy"))
        foot1_traj = np.load(os.path.join(script_dir, "foot1_trajectory.npy"))
        foot2_traj = np.load(os.path.join(script_dir, "foot2_trajectory.npy"))
        print(f"Successfully loaded trajectories:")
        print(f"  Base trajectory: {base_traj.shape}")
        print(f"  COM trajectory: {com_traj.shape}")
        print(f"  Foot 1 trajectory: {foot1_traj.shape}")
        print(f"  Foot 2 trajectory: {foot2_traj.shape}")
    except FileNotFoundError as e:
        print(f"Error: Trajectory files not found: {e}")
        print("Please run 'generate_humanoid_gait.py' first.")
        return
    
    # Load the model
    try:
        model = load_model_with_assets()
        data = mujoco.MjData(model)
        print("Model and data initialized successfully.")
    except Exception as e:
        print(f"Error loading converted MJCF model: {e}")
        print("Ensure 'conversion.py' ran successfully and generated 'legged_robot_ik.xml'.")
        return

    # Adjust timestep for smoother playback
    model.opt.timestep = 0.001
    
    # Enable contacts for realistic physics
    print("Enabling contacts to prevent penetration.")
    
    # --- Simulation Setup ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set initial joint positions
        mujoco.mj_resetData(model, data)
        
        # Set camera parameters for a good view - facing the walking robot
        viewer.cam.azimuth = 0       # 0 = face forward along X axis (direction of walking)
        viewer.cam.elevation = -15   # Slightly downward angle
        viewer.cam.distance = 0.4    # Distance from robot
        viewer.cam.lookat[:] = [0.01, 0.0, 0.22]  # Look at center of robot during walking

        # --- Main Loop ---
        step_index = 0
        num_steps = min(len(base_traj), len(com_traj), len(foot1_traj), len(foot2_traj))
        
        print(f"\nStarting IK trajectory simulation ({num_steps} steps)...")
        print(f"Initial qpos: {data.qpos}")
        
        # Get initial site positions for comparison
        com_site_id = model.site(name='com_site').id
        foot1_site_id = model.site(name='foot1_site').id
        foot2_site_id = model.site(name='foot2_site').id
        print(f"Initial site positions:")
        print(f"  COM: {data.site_xpos[com_site_id]}")
        print(f"  Foot1: {data.site_xpos[foot1_site_id]}")
        print(f"  Foot2: {data.site_xpos[foot2_site_id]}")
        
        # Store previous solution for smoothing
        prev_qpos = data.qpos.copy()
        
        while viewer.is_running() and step_index < num_steps:
            step_start = time.time()

            # Get target positions for the current step
            base_target = base_traj[step_index]
            com_target = com_traj[step_index]
            foot1_target = foot1_traj[step_index]
            foot2_target = foot2_traj[step_index]

            # Solve IK to find the required joint angles
            try:
                qpos_solution, success = compute_ik_solution(
                    model, data, 
                    base_target,
                    com_target, 
                    foot1_target, 
                    foot2_target,
                    max_iterations=50,
                    tolerance=0.002
                )
                
                # Debug: show qpos change and current site positions
                qpos_change = np.linalg.norm(qpos_solution - data.qpos)
                
                if not success:
                    pass  # IK may not have converged, but we still use the best solution
                
                # Smooth the solution with previous frame (low-pass filter to reduce jitter)
                # Blend 95% new + 5% previous (mostly new to reach targets quickly)
                smoothed_qpos = 0.95 * qpos_solution + 0.05 * prev_qpos
                
                # Set the robot's state to the smoothed solution
                data.qpos[:] = smoothed_qpos
                mujoco.mj_forward(model, data)
                
                # Save for next iteration
                prev_qpos = smoothed_qpos.copy()
                
                # Show sites after update
                if step_index % 100 == 0:
                    print(f"Step {step_index}: qpos_change={qpos_change:.6f}")
                    print(f"  Target COM: {com_target}, Actual: {data.site_xpos[com_site_id]}")
                    print(f"  New qpos: {qpos_solution}")
                
            except Exception as e:
                print(f"\nError during IK at step {step_index}: {e}")
                import traceback
                traceback.print_exc()
                print(f"Stopping simulation.")
                break

            # Update the viewer
            viewer.sync()
            
            if step_index % 50 == 0 and step_index > 0:
                print(f"Step {step_index}/{num_steps}")
            
            step_index += 1

            # Slow down playback (10x slower for viewing)
            time_until_next_step = model.opt.timestep * 10 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        print(f"Trajectory finished. Total steps: {step_index}")
        
        # Keep viewer open after trajectory finishes
        while viewer.is_running():
            viewer.sync()

if __name__ == "__main__":
    main()
