#!/usr/bin/env python3
"""
Test: Both feet firmly on sand with symmetric gait
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
    
    # CRITICAL: Make gait SYMMETRIC - both feet same height at all times
    # Use LEFT leg motions for BOTH legs (mirror for right)
    amplitude_scale = 0.30  # Moderate amplitude to push sand effectively
    ik_left_hip = ik_left_hip * amplitude_scale
    ik_left_knee = ik_left_knee * amplitude_scale
    ik_left_ankle = ik_left_ankle * amplitude_scale
    
    # Right leg MIRRORS left (same motions)
    ik_right_hip = ik_left_hip.copy()
    ik_right_knee = ik_left_knee.copy()
    ik_right_ankle = ik_left_ankle.copy()
    
    # Interpolations
    interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
    interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
    interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
    interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
    interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
    interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')
    
    # Control gains - increased for stronger sand pushing
    Kp = 800.0      # Stronger control to push sand
    Kd = 80.0
    gait_period = 3.0
    
    # Get body IDs
    try:
        foot_left_id = model.body("foot_1").id
        foot_right_id = model.body("foot_2").id
    except:
        foot_left_id = None
        foot_right_id = None
    
    hip_id = model.body("hip").id
    
    print("=" * 100)
    print("SYMMETRIC GAIT TEST - BOTH FEET ON SAND")
    print("=" * 100)
    print(f"Strategy: Symmetric leg motions (15% amplitude) + forward push")
    print(f"Joint control: Kp={Kp}, Kd={Kd}")
    print("=" * 100)
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
            
            # SYMMETRIC: Both legs follow same trajectory (no phase offset)
            data.ctrl[0] = Kp * (interp_left_hip(t_cycle) - data.qpos[3]) - Kd * data.qvel[3]
            data.ctrl[1] = Kp * (interp_left_knee(t_cycle) - data.qpos[4]) - Kd * data.qvel[4]
            data.ctrl[2] = Kp * (interp_left_ankle(t_cycle) - data.qpos[5]) - Kd * data.qvel[5]
            data.ctrl[3] = Kp * (interp_right_hip(t_cycle) - data.qpos[6]) - Kd * data.qvel[6]
            data.ctrl[4] = Kp * (interp_right_knee(t_cycle) - data.qpos[7]) - Kd * data.qvel[7]
            data.ctrl[5] = Kp * (interp_right_ankle(t_cycle) - data.qpos[8]) - Kd * data.qvel[8]
            
            # Forward force - increased to push sand and move forward
            forward_force = 3000.0
            data.xfrc_applied[0, 0] = forward_force
            
            # Rotation control
            rotation_kp = 30.0
            rotation_kd = 15.0
            torque_z = rotation_kp * (0.0 - data.qpos[2]) - rotation_kd * data.qvel[2]
            data.xfrc_applied[0, 5] = torque_z
            
            # Lateral control
            lateral_kp = 80.0
            lateral_kd = 15.0
            lateral_force = lateral_kp * (0.0 - data.qpos[1]) - lateral_kd * data.qvel[1]
            data.xfrc_applied[0, 1] = lateral_force
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            step_count += 1
            
            if step_count % 100 == 0:
                hip_x = data.xpos[hip_id][0]
                hip_y = data.xpos[hip_id][1]
                
                left_z = data.xpos[foot_left_id][2] if foot_left_id else 0
                right_z = data.xpos[foot_right_id][2] if foot_right_id else 0
                
                distance = hip_x - 0.150
                status_left = "ON" if 0.442 <= left_z <= 0.460 else "OFF"
                status_right = "ON" if 0.442 <= right_z <= 0.460 else "OFF"
                
                print(f"T:{data.time:6.2f}s | Hip X:{hip_x:+7.4f}m | Dist:{distance:+7.4f}m | Feet Z: L={left_z:.4f}m ({status_left:3s}) | R={right_z:.4f}m ({status_right:3s})")

if __name__ == "__main__":
    main()
