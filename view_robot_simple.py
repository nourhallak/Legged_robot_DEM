#!/usr/bin/env python3
"""
Simple robot walking visualization - should show robot clearly
"""

import numpy as np
import mujoco
import mujoco.viewer
from scipy.interpolate import interp1d

def main():
    # Load model with sand
    model = mujoco.MjModel.from_xml_path("legged_robot_sand_shifted.xml")
    data = mujoco.MjData(model)
    
    # Load IK trajectories
    ik_times = np.load("ik_times.npy")
    ik_left_hip = np.load("ik_left_hip.npy")
    ik_left_knee = np.load("ik_left_knee.npy")
    ik_left_ankle = np.load("ik_left_ankle.npy")
    ik_right_hip = np.load("ik_right_hip.npy")
    ik_right_knee = np.load("ik_right_knee.npy")
    ik_right_ankle = np.load("ik_right_ankle.npy")
    
    # Load full trajectory data
    traj_times = np.load("traj_times.npy")
    traj_base_pos = np.load("traj_base_pos_shifted.npy")  # Use shifted trajectory (offset to sand position)
    
    # Interpolations
    interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
    interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
    interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
    interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
    interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
    interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')
    
    # Base position interpolation from full trajectory (traj_times has 10000 points)
    interp_base_x = interp1d(traj_times, traj_base_pos[:, 0], kind='cubic', fill_value='extrapolate')
    interp_base_y = interp1d(traj_times, traj_base_pos[:, 1], kind='cubic', fill_value='extrapolate')
    interp_base_z = interp1d(traj_times, traj_base_pos[:, 2], kind='cubic', fill_value='extrapolate')
    
    # Control gains
    Kp, Kd = 1000.0, 100.0
    gait_period = 3.0
    traj_period = traj_times.max()  # 100.0 seconds
    
    # Base motion control (high gains for forward walking)
    base_Kp = 5000.0
    base_Kd = 50.0
    
    # Get body IDs
    hip_id = model.body("hip").id
    
    print("=" * 60)
    print("ROBOT WALKING VIEWER")
    print("=" * 60)
    print(f"Robot hip body loaded: {model.body('hip').name}")
    print(f"Total sand particles: {model.nbody - 7}")  # 1000 sand + 7 robot bodies
    print(f"Meshes in model: {model.nmesh}")
    print()
    print("ROBOT POSITION: Connected and touching sand")
    print("Sand location: X=[0.150, 0.482]m")
    print("Robot hip starts at: X=0.150m (beginning of sand)")
    print("=" * 60)
    print("Robot should be clearly visible walking on sand!")
    print("Close window to end simulation")
    print("=" * 60)
    print()
    
    step_count = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Time in gait cycle (repeats every 3 seconds)
            t_cycle = data.time % gait_period
            
            # Time in full trajectory (for base position - scales linearly with time)
            # Map current time to trajectory time (0 to 100 seconds, then repeat)
            t_traj = data.time % traj_period
            
            # Apply joint control (repeating 3-second gait)
            data.ctrl[0] = Kp * (interp_left_hip(t_cycle) - data.qpos[3]) - Kd * data.qvel[3]
            data.ctrl[1] = Kp * (interp_left_knee(t_cycle) - data.qpos[4]) - Kd * data.qvel[4]
            data.ctrl[2] = Kp * (interp_left_ankle(t_cycle) - data.qpos[5]) - Kd * data.qvel[5]
            data.ctrl[3] = Kp * (interp_right_hip(t_cycle) - data.qpos[6]) - Kd * data.qvel[6]
            data.ctrl[4] = Kp * (interp_right_knee(t_cycle) - data.qpos[7]) - Kd * data.qvel[7]
            data.ctrl[5] = Kp * (interp_right_ankle(t_cycle) - data.qpos[8]) - Kd * data.qvel[8]
            
            # Apply forward force from full trajectory
            target_base_x = interp_base_x(t_traj)
            forward_force = base_Kp * (target_base_x - data.qpos[0]) - base_Kd * data.qvel[0]
            data.xfrc_applied[0, 0] = forward_force
            
            # Apply rotational damping to prevent spinning
            rotation_damping = 50.0
            data.xfrc_applied[0, 5] = -rotation_damping * data.qvel[2]
            
            # Lateral stability
            data.xfrc_applied[0, 1] = 2500.0 * (0.0 - data.qpos[1]) - 150.0 * data.qvel[1]
            
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            
            step_count += 1
            
            # Print status every 100 steps (0.5 seconds)
            if step_count % 100 == 0:
                hip_x = data.xpos[hip_id][0]
                t = data.time
                print(f"Time: {t:7.2f}s | Robot X: {hip_x:+8.4f}m | Steps: {step_count}")
            
            # Continue forever - close window to exit

if __name__ == "__main__":
    main()
