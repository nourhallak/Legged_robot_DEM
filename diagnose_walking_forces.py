#!/usr/bin/env python3
"""
Diagnose why robot is not moving despite large forward force
Check: force direction, magnitude, reaction forces, contacts
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
    ik_left_hip = np.load("ik_left_hip.npy") * 0.25
    ik_left_knee = np.load("ik_left_knee.npy") * 0.25
    ik_left_ankle = np.load("ik_left_ankle.npy") * 0.25
    ik_right_hip = np.load("ik_right_hip.npy") * 0.25
    ik_right_knee = np.load("ik_right_knee.npy") * 0.25
    ik_right_ankle = np.load("ik_right_ankle.npy") * 0.25
    
    interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
    interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
    interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
    interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
    interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
    interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')
    
    Kp = 1000.0
    Kd = 100.0
    gait_period = 3.0
    
    hip_id = model.body("hip").id
    foot_left_id = model.body("foot_1").id
    foot_right_id = model.body("foot_2").id
    
    print("=" * 120)
    print("DIAGNOSE WALKING FORCES")
    print("=" * 120)
    print(f"Total bodies in model: {model.nbody}")
    print(f"Hip body name='hip' ID: {hip_id}")
    print(f"Left foot body name='foot_1' ID: {foot_left_id}")
    print(f"Right foot body name='foot_2' ID: {foot_right_id}")
    
    # Print first 10 body names
    print("\nFirst 10 body names in model:")
    for i in range(min(10, model.nbody)):
        print(f"  ID {i}: {model.body(i).name}")
    print()
    
    step_count = 0
    contact_count_left = 0
    contact_count_right = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.distance = 1.5
        viewer.cam.elevation = 0
        viewer.cam.lookat[:] = [0.3, 0, 0.5]
        
        while viewer.is_running():
            t_cycle = data.time % gait_period
            phase_offset = gait_period * 0.5
            t_left = (data.time + phase_offset) % gait_period
            
            # Joint control
            data.ctrl[3] = Kp * (interp_right_hip(t_cycle) - data.qpos[6]) - Kd * data.qvel[6]
            data.ctrl[4] = Kp * (interp_right_knee(t_cycle) - data.qpos[7]) - Kd * data.qvel[7]
            data.ctrl[5] = Kp * (interp_right_ankle(t_cycle) - data.qpos[8]) - Kd * data.qvel[8]
            data.ctrl[0] = Kp * (interp_left_hip(t_left) - data.qpos[3]) - Kd * data.qvel[3]
            data.ctrl[1] = Kp * (interp_left_knee(t_left) - data.qpos[4]) - Kd * data.qvel[4]
            data.ctrl[2] = Kp * (interp_left_ankle(t_left) - data.qpos[5]) - Kd * data.qvel[5]
            
            # FORWARD FORCE - test different magnitudes
            forward_force = 6000.0 if data.time < 20.0 else 10000.0
            data.xfrc_applied[hip_id, 0] = forward_force
            
            # Rotation and lateral control
            rotation_kp = 40.0
            rotation_kd = 20.0
            torque_z = rotation_kp * (0.0 - data.qpos[2]) - rotation_kd * data.qvel[2]
            data.xfrc_applied[hip_id, 5] = torque_z
            
            lateral_kp = 100.0
            lateral_kd = 20.0
            lateral_force = lateral_kp * (0.0 - data.qpos[1]) - lateral_kd * data.qvel[1]
            data.xfrc_applied[hip_id, 1] = lateral_force
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            step_count += 1
            
            if step_count % 100 == 0:
                hip_x = data.xpos[hip_id][0]
                hip_vel_x = data.cvel[hip_id, 3]  # linear velocity in X
                hip_vel = np.linalg.norm(data.cvel[hip_id, 3:6])  # total velocity
                
                left_z = data.xpos[foot_left_id][2]
                right_z = data.xpos[foot_right_id][2]
                
                distance = hip_x - 0.150
                status_left = "ON" if 0.442 <= left_z <= 0.460 else "OFF"
                status_right = "ON" if 0.442 <= right_z <= 0.460 else "OFF"
                
                # Applied force and actual velocity
                applied_frc = data.xfrc_applied[hip_id, 0]
                hip_mass = model.body("hip").mass[0]
                
                print(f"T:{data.time:6.2f}s | Hip X:{hip_x:+7.4f}m | Vel X:{hip_vel_x:+7.4f}m/s (total:{hip_vel:+7.4f}m/s) | " +
                      f"Dist:{distance:+7.4f}m | Frc:{applied_frc:+8.0f}N | Feet Z: L={left_z:.4f}m ({status_left:3s}), R={right_z:.4f}m ({status_right:3s})")

if __name__ == "__main__":
    main()
