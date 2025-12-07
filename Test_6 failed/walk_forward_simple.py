#!/usr/bin/env python3
"""
Robot walking with DIRECT X-position control (simple forward walking)
"""

import numpy as np
import mujoco
import mujoco.viewer
from scipy.interpolate import interp1d

def main():
    # Load model with sand
    model = mujoco.MjModel.from_xml_path("legged_robot_sand_shifted.xml")
    data = mujoco.MjData(model)
    
    # Load IK trajectories (just for leg joint angles)
    ik_times = np.load("ik_times.npy")
    ik_left_hip = np.load("ik_left_hip.npy")
    ik_left_knee = np.load("ik_left_knee.npy")
    ik_left_ankle = np.load("ik_left_ankle.npy")
    ik_right_hip = np.load("ik_right_hip.npy")
    ik_right_knee = np.load("ik_right_knee.npy")
    ik_right_ankle = np.load("ik_right_ankle.npy")
    
    # Interpolations for leg joints
    interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
    interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
    interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
    interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
    interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
    interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')
    
    # Control gains
    Kp, Kd = 1000.0, 100.0
    gait_period = 3.0
    
    # Base motion control - VERY HIGH GAINS for forward pushing
    base_Kp = 10000.0  # INCREASED from 5000
    base_Kd = 100.0    # INCREASED from 50
    
    # Forward walking velocity: 0.02 m/s = 2 cm/s (much slower, more realistic)
    forward_velocity = 0.02
    
    # Get body IDs
    hip_id = model.body("hip").id
    
    print("=" * 80)
    print("ROBOT FORWARD WALKING (DIRECT X-POSITION CONTROL)")
    print("=" * 80)
    print(f"Target forward velocity: {forward_velocity} m/s")
    print(f"Base control gains: Kp={base_Kp}, Kd={base_Kd}")
    print("=" * 80)
    print()
    
    step_count = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.distance = 1.5
        viewer.cam.elevation = 0
        viewer.cam.lookat[:] = [0.3, 0, 0.5]
        
        while viewer.is_running():
            # Time in gait cycle (repeats every 3 seconds)
            t_cycle = data.time % gait_period
            
            # Apply joint control (repeating 3-second gait)
            data.ctrl[0] = Kp * (interp_left_hip(t_cycle) - data.qpos[3]) - Kd * data.qvel[3]
            data.ctrl[1] = Kp * (interp_left_knee(t_cycle) - data.qpos[4]) - Kd * data.qvel[4]
            data.ctrl[2] = Kp * (interp_left_ankle(t_cycle) - data.qpos[5]) - Kd * data.qvel[5]
            data.ctrl[3] = Kp * (interp_right_hip(t_cycle) - data.qpos[6]) - Kd * data.qvel[6]
            data.ctrl[4] = Kp * (interp_right_knee(t_cycle) - data.qpos[7]) - Kd * data.qvel[7]
            data.ctrl[5] = Kp * (interp_right_ankle(t_cycle) - data.qpos[8]) - Kd * data.qvel[8]
            
            # DIRECT X-position control: move forward at constant velocity
            # Target position increases linearly with time
            target_base_x = 0.150 + (forward_velocity * data.time)
            
            # PD control for X position
            forward_force = base_Kp * (target_base_x - data.qpos[0]) - base_Kd * data.qvel[0]
            data.xfrc_applied[0, 0] = forward_force
            
            # Apply rotational damping to prevent spinning
            rotation_damping = 50.0
            data.xfrc_applied[0, 5] = -rotation_damping * data.qvel[2]
            
            # Lateral stability
            data.xfrc_applied[0, 1] = 2500.0 * (0.0 - data.qpos[1]) - 150.0 * data.qvel[1]
            
            # Step physics simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            
            step_count += 1
            
            # Print status every 100 steps (0.5 seconds)
            if step_count % 100 == 0:
                hip_x = data.xpos[hip_id][0]
                target_x = target_base_x
                fwd_force = forward_force
                distance_traveled = hip_x - 0.150
                print(f"Time: {data.time:7.2f}s | Robot X: {hip_x:+8.4f}m | Target X: {target_x:+8.4f}m | Distance: {distance_traveled:+7.4f}m | Force: {fwd_force:+10.1f}N")

if __name__ == "__main__":
    main()
