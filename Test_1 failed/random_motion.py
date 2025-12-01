import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import re

def load_model_with_assets():
    """
    Loads the converted MJCF model ('legged_robot_ik.xml') using the 
    asset loading method to ensure meshes are correctly included.
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

def main():
    try:
        model = load_model_with_assets()
        data = mujoco.MjData(model)
        print("Model and data initialized successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Check if there are any actuators in the model
    if model.nu == 0:
        print("Error: No actuators found in the model.")
        print("Please run the updated 'conversion.py' to add actuators to your XML file.")
        return

    # --- Enable contacts to prevent penetration ---
    print("Enabling contacts to prevent ground and self-penetration.")
    # Don't disable contacts - let them work naturally
    
    # Add damping to stabilize the system
    model.opt.viscosity = 0.1

    # --- Simulation Setup ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset the simulation to the default state
        mujoco.mj_resetData(model, data)
        # Set camera parameters for a good view
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25
        viewer.cam.distance = 0.7
        viewer.cam.lookat[:] = [0.0, 0.0, 0.03]

        # --- Main Loop ---
        # Use a seed for reproducibility
        np.random.seed(42)

        # Store original gravity setting
        original_gravity = model.opt.gravity.copy()
        
        # Set a random target for the controls
        target = np.random.uniform(low=-1.0, high=1.0, size=model.nu)
        
        while viewer.is_running():
            step_start = time.time()

            # Disable gravity for the first 0.5 seconds to allow the robot to settle
            if data.time < 0.5:
                model.opt.gravity[2] = 0
            else:
                model.opt.gravity[:] = original_gravity

            # Every 2 seconds, set a new random target
            if int(data.time) % 2 == 0 and int(data.time) != int(data.time - model.opt.timestep):
                target = np.random.uniform(low=-1.0, high=1.0, size=model.nu)
                print(f"Time: {data.time:.2f}s, New random target: {np.round(target, 2)}")

            # Apply the control signal
            data.ctrl[:] = target

            # Step the simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            # Slow down simulation for better viewing (10x slower)
            time_until_next_step = model.opt.timestep * 10 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()