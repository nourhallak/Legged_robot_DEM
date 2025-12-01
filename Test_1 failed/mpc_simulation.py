import mujoco
import mujoco.viewer
from mujoco import quat_mul, quat_conjugate
import numpy as np
import time
import os

# --- MPC Controller (Conceptual) ---

def mpc_controller(model, data, com_ref, hip_orn_ref, foot1_ref, foot2_ref):
    """
    A placeholder for a Model Predictive Control solver.

    In a real implementation, this function would:
    1.  Define a cost function based on tracking errors for CoM, hip orientation,
        and foot positions, while also penalizing high torques and instability (ZMP).
    2.  Use a predictive model of the robot's dynamics.
    3.  Call an optimization solver (like OSQP, IPOPT, or a custom one) to find the
        optimal sequence of control inputs (torques) over a future time horizon.
    4.  Return the first control input from that optimal sequence.

    For this example, we will implement a simple Proportional-Derivative (PD)
    controller as a stand-in for the MPC output. This will apply torques to
    try and match a target `qpos` derived from Inverse Kinematics.
    """
    
    # --- Stand-in for MPC: A feedback controller with multiple components ---

    # -- Component 1: Leg PD Control (based on IK) --
    # This part tries to move the legs to follow the pre-planned trajectory.
    try:
        qpos_target = solve_ik_for_targets(model, data, com_ref, foot1_ref, foot2_ref)
    except ValueError as e:
        print(f"IK failed: {e}")
        return np.zeros(model.nu) # Return zero torques if IK fails

    # Gains for the leg joints
    kp_leg = 50.0  # Proportional gain (stiffness)
    kv_leg = 5.0   # Derivative gain (damping)
    
    # We only control the 6 actuated joints, not the freejoint of the hip
    qpos_actuated = data.qpos[7:]
    qvel_actuated = data.qvel[6:]
    qpos_target_actuated = qpos_target[7:]

    leg_pd_torque = kp_leg * (qpos_target_actuated - qpos_actuated) - kv_leg * qvel_actuated

    # -- Component 2: Hip Orientation Control (Active Stabilization) --
    # This part actively tries to keep the hip horizontal.
    kp_hip = 200.0 # Higher gain to prioritize stability
    kv_hip = 25.0

    # Get current hip orientation and angular velocity
    hip_quat = data.qpos[3:7]
    hip_ang_vel = data.qvel[3:6]

    # Calculate orientation error
    # error_quat = reference_quat * conjugate(current_quat)
    error_quat = quat_mul(hip_orn_ref, quat_conjugate(hip_quat))
    
    # The vector part of the error quaternion is a good approximation of axis-angle error
    orientation_error = error_quat[1:] 
    
    # PD control on orientation error
    hip_stabilization_torque = kp_hip * orientation_error - kv_hip * hip_ang_vel

    # -- Map hip torques to leg joints --
    # We can't directly torque the hip. We must use the legs.
    # This is a simplification. A real controller uses the Jacobian.
    # Here, we apply corrective torques to the hip joints of each leg.
    # Roll error (x-axis) -> torque hip joints
    # Pitch error (y-axis) -> torque knee joints (approx)
    leg_torques_from_hip = np.zeros(6)
    leg_torques_from_hip[[0, 3]] = hip_stabilization_torque[0] # Roll correction
    leg_torques_from_hip[[1, 4]] = -hip_stabilization_torque[1] # Pitch correction

    # -- Component 3: Gravity Compensation --
    # data.qfrc_bias contains Coriolis, centrifugal, and gravity forces.
    gravity_comp_torque = data.qfrc_bias[6:] # Only for actuated joints

    # -- Final Control Signal --
    # Combine all torque components
    ctrl = leg_pd_torque + leg_torques_from_hip + gravity_comp_torque
    
    # Clamp control to actuator limits
    actuator_limits = model.actuator_ctrlrange
    ctrl = np.clip(ctrl, actuator_limits[:, 0], actuator_limits[:, 1])

    return ctrl

def solve_ik_for_targets(model, data, com_target, foot1_target, foot2_target):
    """Helper function to solve IK for a set of targets."""
    com_site_id = model.site('com_site').id
    foot1_site_id = model.site('foot1_site').id
    foot2_site_id = model.site('foot2_site').id

    target_pos = np.array([com_target, foot1_target, foot2_target]).flatten()
    body_ids = np.array([model.site_bodyid[com_site_id], model.site_bodyid[foot1_site_id], model.site_bodyid[foot2_site_id]], dtype=np.int32)
    pos_on_bodies = np.array([model.site_pos[com_site_id], model.site_pos[foot1_site_id], model.site_pos[foot2_site_id]]).flatten()
    
    # Use the current qpos as the initial guess for the IK solver
    qpos_sol = data.qpos.copy()
    
    # Create a temporary data object for the solver to avoid corrupting the main one
    temp_data = mujoco.MjData(model)
    temp_data.qpos[:] = qpos_sol

    # Run the solver
    # Note: mj_solveMGD is deprecated. Using the recommended mj_kinematics and mj_jac functions
    # would be more modern, but MGD is simpler for this direct targetting.
    res = mujoco.mj_solveMGD(model, temp_data, qpos_sol, body_ids, pos_on_bodies, target_pos, 1e-4, 500, None, None)

    if res < 0:
        raise ValueError("IK solver failed to converge.")
    
    return qpos_sol


def main():
    # --- Load Model and Trajectories ---
    # Run conversion.py to generate the model with actuators and sites
    try:
        os.system('python conversion.py')
        print("Conversion script executed successfully.")
    except Exception as e:
        print(f"Failed to run conversion.py: {e}")
        return

    mjcf_path = "legged_robot_ik.xml"
    if not os.path.exists(mjcf_path):
        print(f"Error: MJCF file not found at '{mjcf_path}'")
        return

    try:
        com_traj = np.load("com_trajectory.npy")
        hip_orn_traj = np.load("hip_orientation_trajectory.npy")
        foot1_traj = np.load("foot1_trajectory.npy")
        foot2_traj = np.load("foot2_trajectory.npy")
        print("Successfully loaded all trajectories.")
    except FileNotFoundError as e:
        print(f"Error: Trajectory file not found. {e}")
        print("Please run 'trajectory_planning.py' first.")
        return

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    # --- Simulation Setup ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        
        # Set initial position to be slightly above the ground
        # This is important for dynamic simulations to avoid initial penetrations.
        data.qpos[2] = 0.1
        mujoco.mj_forward(model, data)

        # Set camera
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25
        viewer.cam.distance = 0.7
        viewer.cam.lookat[:] = [0.0, 0.0, 0.03]

        # --- Main Loop ---
        step_index = 0
        num_steps = len(com_traj)
        
        while viewer.is_running() and step_index < num_steps:
            start_time = time.time()

            # Get reference trajectories for the current step
            com_ref = com_traj[step_index]
            hip_orn_ref = hip_orn_traj[step_index]
            foot1_ref = foot1_traj[step_index]
            foot2_ref = foot2_traj[step_index]

            # --- MPC Control Step ---
            # In a real scenario, the MPC would take the current state (data.qpos, data.qvel)
            # and the reference trajectories to compute the optimal torques.
            # Here, we use our PD controller stand-in.
            
            # First, compute inverse dynamics to get gravity and Coriolis forces
            # This is needed for our gravity compensation term.
            mujoco.mj_inverse(model, data) 
            
            # Get control torques from the controller
            torques = mpc_controller(model, data, com_ref, hip_orn_ref, foot1_ref, foot2_ref)
            
            # Apply the torques to the actuators
            data.ctrl[:] = torques

            # Step the dynamic simulation
            mujoco.mj_step(model, data)

            # Update the viewer
            viewer.sync()
            step_index += 1

            # Real-time synchronization
            time_until_next_step = model.opt.timestep - (time.time() - start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        print("Trajectory finished.")
        while viewer.is_running():
            viewer.sync()

if __name__ == "__main__":
    main()