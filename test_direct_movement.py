#!/usr/bin/env python3
"""
Simple walking test: directly move the base position while keeping feet on sand
This validates that the feet can contact sand properly
"""

import numpy as np
import mujoco
import mujoco.viewer
from scipy.interpolate import interp1d

def main():
    # Load model
    model = mujoco.MjModel.from_xml_path("legged_robot_sand_shifted.xml")
    data = mujoco.MjData(model)
    
    # Load IK trajectories
    ik_times = np.load("ik_times.npy")
    ik_left_hip = np.load("ik_left_hip.npy") * 0.10  # 10% amplitude for small movements
    ik_left_knee = np.load("ik_left_knee.npy") * 0.10
    ik_left_ankle = np.load("ik_left_ankle.npy") * 0.10
    ik_right_hip = np.load("ik_right_hip.npy") * 0.10
    ik_right_knee = np.load("ik_right_knee.npy") * 0.10
    ik_right_ankle = np.load("ik_right_ankle.npy") * 0.10
    
    interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
    interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
    interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
    interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
    interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
    interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')
    
    Kp = 50.0       # Control for joint trajectories
    Kd = 5.0
    gait_period = 3.0
    
    hip_id = model.body("hip").id
    foot_left_id = model.body("foot_1").id
    foot_right_id = model.body("foot_2").id
    
    print("=" * 100)
    print("SIMPLE WALKING TEST - DIRECT BASE MOVEMENT")
    print("=" * 100)
    print(f"Strategy: Directly move base X position while keeping feet on sand")
    print(f"Gait amplitude: 10% IK trajectory")
    print(f"Joint control: Kp={Kp}, Kd={Kd}")
    print(f"Forward force: 200-800N (gradual increase)")
    print("=" * 100)
    print()
    
    step_count = 0
    base_x_target = 0.150  # Start here
    base_x_increment = 0.0001  # Move forward very slowly per step
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.distance = 1.5
        viewer.cam.elevation = 0
        viewer.cam.lookat[:] = [0.3, 0, 0.5]
        
        while viewer.is_running():
            t_cycle = data.time % gait_period
            phase_offset = gait_period * 0.5
            t_left = (data.time + phase_offset) % gait_period
            
            # Joint control - follow IK trajectories
            data.ctrl[3] = Kp * (interp_right_hip(t_cycle) - data.qpos[6]) - Kd * data.qvel[6]
            data.ctrl[4] = Kp * (interp_right_knee(t_cycle) - data.qpos[7]) - Kd * data.qvel[7]
            data.ctrl[5] = Kp * (interp_right_ankle(t_cycle) - data.qpos[8]) - Kd * data.qvel[8]
            data.ctrl[0] = Kp * (interp_left_hip(t_left) - data.qpos[3]) - Kd * data.qvel[3]
            data.ctrl[1] = Kp * (interp_left_knee(t_left) - data.qpos[4]) - Kd * data.qvel[4]
            data.ctrl[2] = Kp * (interp_left_ankle(t_left) - data.qpos[5]) - Kd * data.qvel[5]
            
            # APPLY FORWARD FORCE to the hip body
            # Gradually increase force over time
            if data.time < 10.0:
                forward_force = 200.0
            elif data.time < 20.0:
                forward_force = 400.0
            elif data.time < 30.0:
                forward_force = 600.0
            else:
                forward_force = 800.0
            data.xfrc_applied[hip_id, 0] = forward_force
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            step_count += 1
            
            if step_count % 100 == 0:
                hip_x = data.xpos[hip_id][0]
                
                left_z = data.xpos[foot_left_id][2]
                right_z = data.xpos[foot_right_id][2]
                
                distance = hip_x - 0.150
                status_left = "ON" if 0.442 <= left_z <= 0.460 else "OFF"
                status_right = "ON" if 0.442 <= right_z <= 0.460 else "OFF"
                
                print(f"T:{data.time:6.2f}s | Hip X:{hip_x:+7.4f}m | Dist:{distance:+7.4f}m | Feet Z: L={left_z:.4f}m ({status_left:3s}), R={right_z:.4f}m ({status_right:3s})")

if __name__ == "__main__":
    main()
