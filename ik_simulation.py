import mujoco
import mujoco.viewer
import numpy as np
import time
import os

def solve_ik(model, data, com_target_pos, foot1_target_pos, foot2_target_pos):
    """
    Solves inverse kinematics for the robot.

    Args:
        model: The MuJoCo model.
        data: The MuJoCo data.
        com_target_pos (np.ndarray): The target position for the center of mass.
        foot1_target_pos (np.ndarray): The target position for the first foot.
        foot2_target_pos (np.ndarray): The target position for the second foot.

    Returns:
        np.ndarray: The solved joint positions (qpos).
    """
    # IDs of the sites we want to control
    com_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hip")
    foot1_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_1")
    foot2_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_2")

    if com_site_id == -1 or foot1_site_id == -1 or foot2_site_id == -1:
        raise ValueError("Could not find 'com_site', 'foot1_site', or 'foot2_site' in the model. Check your URDF file.")

    # --- IK using mj_solveMGD ---
    # This is a low-level, iterative solver. It finds qpos that minimizes the
    # distance between site positions and their targets.

    # Solver parameters
    # See https://mujoco.readthedocs.io/en/latest/APIreference.html#mjvoption
    # for a description of the flags.
    flags = np.zeros(10, dtype=np.float64)
    
    # Target positions and quaternions for the sites
    # We only care about position, so quaternion can be None (ignored).
    target_pos = np.array([com_target_pos, foot1_target_pos, foot2_target_pos]).flatten()
    target_quat = None

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
    pos_on_bodies = np.array([pos_com, pos_foot1, pos_foot2]).flatten()

    # Initial guess for qpos (current state)
    qpos_sol = data.qpos.copy()

    # Run the solver
    result = mujoco.mj_solveMGD(model, data, qpos_sol, body_ids, pos_on_bodies, target_pos, 1e-6, flags, target_quat)

    if result < 0:
        print("Warning: IK solver failed to converge.")
    
    return qpos_sol


def main():
    # --- Load Model and Trajectories ---
    urdf_path = os.path.join("Legged_robot", "urdf", "Legged_robot.urdf")
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found at '{urdf_path}'")
        return

    try:
        com_traj = np.load("com_trajectory.npy")
        foot1_traj = np.load("foot1_trajectory.npy")
        foot2_traj = np.load("foot2_trajectory.npy")
        print("Successfully loaded all trajectories.")
    except FileNotFoundError:
        print("Error: Trajectory files not found.")
        print("Please run 'trajectory_planning.py' first.")
        return

    model = mujoco.MjModel.from_xml_path(urdf_path)
    data = mujoco.MjData(model)

    # Adjust timestep for smoother playback
    model.opt.timestep = 0.01

    # --- Simulation Setup ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set initial joint positions
        mujoco.mj_resetData(model, data)
        
        # Set camera
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
            qpos_solution = solve_ik(model, data, com_target, foot1_target, foot2_target)

            # Set the robot's state to the IK solution
            # This is a kinematic simulation; we are directly setting the joint angles
            # rather than applying forces and simulating dynamics.
            data.qpos[:] = qpos_solution
            mujoco.mj_forward(model, data)

            # Update the viewer
            viewer.sync()
            step_index += 1

            # Rudimentary real-time synchronization
            time_until_next_step = model.opt.timestep - (time.time() - start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        print("Trajectory finished.")
        # Keep viewer open after trajectory finishes
        while viewer.is_running():
            viewer.sync()

if __name__ == "__main__":
    main()