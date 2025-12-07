#!/usr/bin/env python3
"""
Visualize robot walking forward on sand with MuJoCo viewer
"""

import numpy as np
import mujoco
import mujoco.viewer
from scipy.interpolate import interp1d

def main():
    # Load model with sand in forward region
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
    
    # Interpolations
    interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
    interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
    interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
    interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
    interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
    interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')
    
    # Control parameters
    Kp, Kd = 1000.0, 100.0
    gait_period = 3.0
    
    # Get body IDs
    hip_id = model.body("hip").id
    
    print("Opening MuJoCo viewer - robot walking forward on sand")
    print("Close viewer window to exit\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Time in gait cycle
            t_cycle = data.time % gait_period
            
            # JOINT CONTROL
            data.ctrl[0] = Kp * (interp_left_hip(t_cycle) - data.qpos[3]) - Kd * data.qvel[3]
            data.ctrl[1] = Kp * (interp_left_knee(t_cycle) - data.qpos[4]) - Kd * data.qvel[4]
            data.ctrl[2] = Kp * (interp_left_ankle(t_cycle) - data.qpos[5]) - Kd * data.qvel[5]
            data.ctrl[3] = Kp * (interp_right_hip(t_cycle) - data.qpos[6]) - Kd * data.qvel[6]
            data.ctrl[4] = Kp * (interp_right_knee(t_cycle) - data.qpos[7]) - Kd * data.qvel[7]
            data.ctrl[5] = Kp * (interp_right_ankle(t_cycle) - data.qpos[8]) - Kd * data.qvel[8]
            
            # BASE CONTROL - Lateral stability only
            data.xfrc_applied[0, 1] = 2500.0 * (0.0 - data.qpos[1]) - 150.0 * data.qvel[1]
            
            mujoco.mj_step(model, data)
            
            # Print status
            hip_x = data.xpos[hip_id][0]
            if data.time % 5 < 0.05:  # Every 5s
                print(f"Time: {data.time:6.1f}s | Robot X: {hip_x:+.4f}m")
            
            viewer.sync()
            
            if data.time > 30:
                break

if __name__ == "__main__":
    main()
