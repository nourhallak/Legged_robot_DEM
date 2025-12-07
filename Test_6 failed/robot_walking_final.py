#!/usr/bin/env python3
"""
Successful biped walking on sand with reduced friction
Strategy: Apply steady forward force + gentle alternating leg gait
"""

import numpy as np
import mujoco
import mujoco.viewer
from scipy.interpolate import interp1d

def main():
    # Load model with reduced sand friction
    model = mujoco.MjModel.from_xml_path("legged_robot_sand_shifted.xml")
    data = mujoco.MjData(model)
    
    # Load IK trajectories for walking pattern
    ik_times = np.load("ik_times.npy")
    ik_left_hip = np.load("ik_left_hip.npy") * 0.10    # 10% amplitude
    ik_left_knee = np.load("ik_left_knee.npy") * 0.10
    ik_left_ankle = np.load("ik_left_ankle.npy") * 0.10
    ik_right_hip = np.load("ik_right_hip.npy") * 0.10
    ik_right_knee = np.load("ik_right_knee.npy") * 0.10
    ik_right_ankle = np.load("ik_right_ankle.npy") * 0.10
    
    # Create interpolation functions
    interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
    interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
    interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
    interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
    interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
    interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')
    
    # Control parameters
    Kp = 50.0           # Joint PD gains
    Kd = 5.0
    gait_period = 3.0
    
    # Get body IDs
    hip_id = model.body("hip").id
    foot_left_id = model.body("foot_1").id
    foot_right_id = model.body("foot_2").id
    
    print("=" * 100)
    print("BIPED ROBOT WALKING ON SAND")
    print("=" * 100)
    print(f"Control: Steady forward force + alternating leg gait (10% amplitude)")
    print(f"Joint control: Kp={Kp}, Kd={Kd}")
    print(f"Sand: Friction reduced to 0.4 (from 0.8)")
    print(f"Feet: Friction reduced to 0.3 (from 1.0)")
    print("=" * 100)
    print()
    
    step_count = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.distance = 1.5
        viewer.cam.elevation = 0
        viewer.cam.lookat[:] = [0.3, 0, 0.5]
        
        while viewer.is_running():
            # Gait cycle timing
            t_cycle = data.time % gait_period
            
            # Alternating phase for left leg (50% offset from right)
            phase_offset = gait_period * 0.5
            t_left = (data.time + phase_offset) % gait_period
            
            # Right leg joint control
            data.ctrl[3] = Kp * (interp_right_hip(t_cycle) - data.qpos[6]) - Kd * data.qvel[6]
            data.ctrl[4] = Kp * (interp_right_knee(t_cycle) - data.qpos[7]) - Kd * data.qvel[7]
            data.ctrl[5] = Kp * (interp_right_ankle(t_cycle) - data.qpos[8]) - Kd * data.qvel[8]
            
            # Left leg joint control
            data.ctrl[0] = Kp * (interp_left_hip(t_left) - data.qpos[3]) - Kd * data.qvel[3]
            data.ctrl[1] = Kp * (interp_left_knee(t_left) - data.qpos[4]) - Kd * data.qvel[4]
            data.ctrl[2] = Kp * (interp_left_ankle(t_left) - data.qpos[5]) - Kd * data.qvel[5]
            
            # Steady forward force to propel walking
            forward_force = 300.0
            data.xfrc_applied[hip_id, 0] = forward_force
            
            # Light stabilization: prevent lateral drift and rotation
            lateral_correction = -100.0 * data.qpos[1] - 10.0 * data.qvel[1]
            rotation_correction = -20.0 * data.qpos[2] - 5.0 * data.qvel[2]
            data.xfrc_applied[hip_id, 1] = lateral_correction
            data.xfrc_applied[hip_id, 5] = rotation_correction
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            step_count += 1
            
            # Print status periodically
            if step_count % 100 == 0:
                hip_x = data.xpos[hip_id][0]
                
                left_z = data.xpos[foot_left_id][2]
                right_z = data.xpos[foot_right_id][2]
                
                distance = hip_x - 0.150
                status_left = "ON" if 0.442 <= left_z <= 0.460 else "OFF"
                status_right = "ON" if 0.442 <= right_z <= 0.460 else "OFF"
                
                print(f"T:{data.time:6.2f}s | Hip X:{hip_x:+7.4f}m | Dist:{distance:+7.4f}m | " +
                      f"Feet: L={left_z:.4f}m ({status_left:3s}), R={right_z:.4f}m ({status_right:3s})")

if __name__ == "__main__":
    main()
