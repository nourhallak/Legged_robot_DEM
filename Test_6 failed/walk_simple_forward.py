#!/usr/bin/env python3
"""
Simple forward walking with reduced gait amplitude to prevent backward sliding
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
    
    # Scale down joint angles to reduce aggressive leg motions (prevent backward push)
    amplitude_scale = 0.35  # Use only 35% of original amplitudes (tested working)
    ik_left_hip = ik_left_hip * amplitude_scale
    ik_left_knee = ik_left_knee * amplitude_scale
    ik_left_ankle = ik_left_ankle * amplitude_scale
    ik_right_hip = ik_right_hip * amplitude_scale
    ik_right_knee = ik_right_knee * amplitude_scale
    ik_right_ankle = ik_right_ankle * amplitude_scale
    
    # Interpolations
    interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
    interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
    interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
    interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
    interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
    interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')
    
    # Control gains - tuned for 35% amplitude (lower gains, more stable)
    Kp = 600.0      # Joint tracking - reduced for stability with lower amplitude
    Kd = 60.0       # Joint damping
    gait_period = 3.0
    
    # Base motion control - push forward steadily
    base_Kp = 1500.0  # Base forward push gain (tuned for 35% amplitude)
    base_Kd = 50.0
    
    # Forward walking: move at 0.015 m/s (1.5 cm/s - steady, stable pace)
    forward_velocity = 0.015
    
    # Get body IDs
    hip_id = model.body("hip").id
    
    print("=" * 80)
    print("SIMPLE FORWARD WALKING - REDUCED GAIT AMPLITUDE")
    print("=" * 80)
    print(f"Gait amplitude scale: {amplitude_scale*100:.0f}%")
    print(f"Forward velocity target: {forward_velocity} m/s")
    print(f"Joint control: Kp={Kp}, Kd={Kd}")
    print(f"Base control: Kp={base_Kp}, Kd={base_Kd}")
    print("=" * 80)
    print()
    
    step_count = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.distance = 1.5
        viewer.cam.elevation = 0
        viewer.cam.lookat[:] = [0.3, 0, 0.5]
        
        while viewer.is_running():
            # Time in gait cycle
            t_cycle = data.time % gait_period
            
            # Apply reduced-amplitude joint control
            data.ctrl[0] = Kp * (interp_left_hip(t_cycle) - data.qpos[3]) - Kd * data.qvel[3]
            data.ctrl[1] = Kp * (interp_left_knee(t_cycle) - data.qpos[4]) - Kd * data.qvel[4]
            data.ctrl[2] = Kp * (interp_left_ankle(t_cycle) - data.qpos[5]) - Kd * data.qvel[5]
            data.ctrl[3] = Kp * (interp_right_hip(t_cycle) - data.qpos[6]) - Kd * data.qvel[6]
            data.ctrl[4] = Kp * (interp_right_knee(t_cycle) - data.qpos[7]) - Kd * data.qvel[7]
            data.ctrl[5] = Kp * (interp_right_ankle(t_cycle) - data.qpos[8]) - Kd * data.qvel[8]
            
            # Direct X position control: move forward at constant velocity
            target_base_x = 0.150 + (forward_velocity * data.time)
            
            # PD control for base X position
            forward_force = base_Kp * (target_base_x - data.qpos[0]) - base_Kd * data.qvel[0]
            data.xfrc_applied[0, 0] = forward_force
            
            # CRITICAL: Keep robot oriented straight (prevent circular walking)
            # Strong rotation control to counteract asymmetric gait
            rotation_kp = 50.0  # Strong rotation correction
            rotation_kd = 20.0
            target_rotation = 0.0  # Keep heading straight
            torque_z = rotation_kp * (target_rotation - data.qpos[2]) - rotation_kd * data.qvel[2]
            data.xfrc_applied[0, 5] = torque_z
            
            # Lateral stability - prevent sideways drift
            lateral_kp = 100.0
            lateral_kd = 20.0
            lateral_force = lateral_kp * (0.0 - data.qpos[1]) - lateral_kd * data.qvel[1]
            data.xfrc_applied[0, 1] = lateral_force
            
            # Step physics
            mujoco.mj_step(model, data)
            viewer.sync()
            
            step_count += 1
            
            if step_count % 100 == 0:
                hip_x = data.xpos[hip_id][0]
                distance = hip_x - 0.150
                print(f"Time: {data.time:7.2f}s | Robot X: {hip_x:+8.4f}m | Distance: {distance:+7.4f}m | Force: {forward_force:+10.1f}N")

if __name__ == "__main__":
    main()
