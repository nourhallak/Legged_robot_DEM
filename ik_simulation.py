import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import re

# --- Utility Functions for Asset Loading (Copied from conversion_fix.py logic) ---

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
    # Look for files referenced by <mesh file="...">
    MESH_PATTERN = r'file="([^"]+\.STL)"'
    all_mesh_files = set(re.findall(MESH_PATTERN, mjcf_content))

    if not all_mesh_files:
        print("Warning: No mesh files found referenced in the converted MJCF.")

    # 4. Build the Assets Dictionary (Reading file content as bytes)
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
    # The string loading method is essential for models converted from URDF 
    # with embedded sites and external assets.
    model = mujoco.MjModel.from_xml_string(mjcf_content, assets=assets)
    return model

# --- Inverse Kinematics Solver ---

def solve_ik(model, data, com_target_pos, foot1_target_pos, foot2_target_pos):
    """
    Solves inverse kinematics for the robot.
    """
    try:
        # Use the model's indexing to get the site IDs
        # If the model load function worked, these sites must exist.
        com_site_id = model.site(name='com_site').id
        foot1_site_id = model.site(name='foot1_site').id
        foot2_site_id = model.site(name='foot2_site').id

        # Check if the IDs are valid (>= 0)
        if com_site_id < 0 or foot1_site_id < 0 or foot2_site_id < 0:
            # This should be unreachable if model.site() succeeded
            raise ValueError("One or more sites were not found by MuJoCo.")

    except Exception as e:
        # Re-raise or handle the error
        # This error now indicates a problem in the converted XML, not the loader.
        raise ValueError(f"Could not find required sites: {e}")
        
    # --- IK using mj_solveMGD ---

    # Solver parameters
    # Flags are typically used for advanced solver control; setting to zero for default behavior
    flags = np.zeros(10, dtype=np.float64)
    
    # Target positions and quaternions for the sites
    # Flatten the array of 3 target positions (3 * 3 = 9 elements)
    target_pos = np.array([com_target_pos, foot1_target_pos, foot2_target_pos]).flatten()
    target_quat = None # We are only solving for position, not orientation

    # Body IDs for the sites
    body_com = model.site_bodyid[com_site_id]
    body_foot1 = model.site_bodyid[foot1_site_id]
    body_foot2 = model.site_bodyid[foot2_site_id]

    # Local position of sites within their bodies
    pos_com = model.site_pos[com_site_id]
    pos_foot1 = model.site_pos[foot1_site_id]
    pos_foot2 = model.site_pos[foot2_site_id]

    # The list of bodies to control
    body_ids = np.array([body_com, body_foot1, body_foot2], dtype=np.int32)
    
    # The local position of the sites on those bodies
    # Flatten the array of 3 local positions (3 * 3 = 9 elements)
    pos_on_bodies = np.array([pos_com, pos_foot1, pos_foot2]).flatten()

    # Initial guess for qpos (current state)
    qpos_sol = data.qpos.copy()

    # Run the solver: Find the joint angles (qpos_sol) that satisfy the pose targets
    result = mujoco.mj_solveMGD(model, data, qpos_sol, body_ids, pos_on_bodies, target_pos, 1e-6, flags, target_quat)

    if result < 0:
        # For a full IK trajectory, it's better to just log a warning 
        # and use the best effort solution, rather than crashing.
        print(f"Warning: IK solver failed to converge at step {data.time}.")
    
    return qpos_sol


def main():
    # --- Load Model and Trajectories ---
    
    try:
        # Load pre-calculated trajectory files
        com_traj = np.load("com_trajectory.npy")
        foot1_traj = np.load("foot1_trajectory.npy")
        foot2_traj = np.load("foot2_trajectory.npy")
        print("Successfully loaded all trajectories.")
    except FileNotFoundError:
        print("Error: Trajectory files not found.")
        print("Please run 'trajectory_planning.py' first.")
        return
    

    # 3. Load the model using the asset loader function
    try:
        model = load_model_with_assets()
        data = mujoco.MjData(model)
        print("Model and data initialized successfully.")
    except Exception as e:
        print(f"Error loading converted MJCF model: {e}")
        print("Ensure 'conversion.py' ran successfully and generated 'legged_robot_ik.xml'.")
        return

    # Adjust timestep for smoother playback
    model.opt.timestep = 0.01
    
    # --- Simulation Setup ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set initial joint positions
        mujoco.mj_resetData(model, data)
        
        # Set camera parameters for a good view of the legged robot
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25
        viewer.cam.distance = 0.3
        viewer.cam.lookat[:] = [0.0, 0.0, 0.03]

        # --- Main Loop ---
        step_index = 0
        num_steps = len(com_traj)
        
        while viewer.is_running() and step_index < num_steps:
            start_time = time.time()

            # Get target positions for the current step
            com_target = com_traj[step_index]
            foot1_target = foot1_traj[step_index]
            foot2_target = foot2_traj[step_index]

            # Solve IK to find the required joint angles (qpos)
            try:
                # The IK solver updates data.qpos internally
                qpos_solution = solve_ik(model, data, com_target, foot1_target, foot2_target)
            except ValueError as e:
                print(f"Fatal error during IK initialization. Stopping simulation: {e}")
                break

            # Set the robot's state to the IK solution
            # The qpos_solution returned by solve_ik is already the solved qpos
            data.qpos[:] = qpos_solution
            # Update forward dynamics to apply the new qpos and compute new site positions
            mujoco.mj_forward(model, data) 

            # Update the viewer
            viewer.sync()
            step_index += 1

            # Rudimentary real-time synchronization to match model.opt.timestep
            time_until_next_step = model.opt.timestep - (time.time() - start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        print("Trajectory finished.")
        # Keep viewer open after trajectory finishes
        while viewer.is_running():
            viewer.sync()

if __name__ == "__main__":
    main()