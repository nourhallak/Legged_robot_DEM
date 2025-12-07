#!/usr/bin/env python3
"""
Diagnose walking issues: movement and foot contact
"""

import numpy as np
import mujoco
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
    
    # Scale amplitude
    amplitude_scale = 0.35
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
    
    # Control gains
    Kp = 600.0
    Kd = 60.0
    base_Kp = 1500.0
    base_Kd = 50.0
    gait_period = 3.0
    forward_velocity = 0.02  # Try higher velocity
    
    rotation_kp = 50.0
    rotation_kd = 20.0
    lateral_kp = 100.0
    lateral_kd = 20.0
    
    # Get body IDs for foot positions
    try:
        foot_left_id = model.body("foot_1").id
        foot_right_id = model.body("foot_2").id
    except:
        foot_left_id = None
        foot_right_id = None
    
    hip_id = model.body("hip").id
    
    print("=" * 100)
    print("DIAGNOSIS: Walking issues - Movement and Foot Contact")
    print("=" * 100)
    print(f"Forward velocity target: {forward_velocity} m/s")
    print(f"Sand position Z range: 0.442-0.450m")
    print("=" * 100)
    print()
    
    step_count = 0
    start_x = data.qpos[0]
    positions = []
    foot_heights = {'left': [], 'right': [], 'time': []}
    
    while data.time < 30.0:
        t_cycle = data.time % gait_period
        
        # Joint control
        data.ctrl[0] = Kp * (interp_left_hip(t_cycle) - data.qpos[3]) - Kd * data.qvel[3]
        data.ctrl[1] = Kp * (interp_left_knee(t_cycle) - data.qpos[4]) - Kd * data.qvel[4]
        data.ctrl[2] = Kp * (interp_left_ankle(t_cycle) - data.qpos[5]) - Kd * data.qvel[5]
        data.ctrl[3] = Kp * (interp_right_hip(t_cycle) - data.qpos[6]) - Kd * data.qvel[6]
        data.ctrl[4] = Kp * (interp_right_knee(t_cycle) - data.qpos[7]) - Kd * data.qvel[7]
        data.ctrl[5] = Kp * (interp_right_ankle(t_cycle) - data.qpos[8]) - Kd * data.qvel[8]
        
        # Base X control - STRONG FORWARD PUSH
        target_base_x = 0.150 + (forward_velocity * data.time)
        forward_force = base_Kp * (target_base_x - data.qpos[0]) - base_Kd * data.qvel[0]
        data.xfrc_applied[0, 0] = forward_force
        
        # Rotation control
        torque_z = rotation_kp * (0.0 - data.qpos[2]) - rotation_kd * data.qvel[2]
        data.xfrc_applied[0, 5] = torque_z
        
        # Lateral control
        lateral_force = lateral_kp * (0.0 - data.qpos[1]) - lateral_kd * data.qvel[1]
        data.xfrc_applied[0, 1] = lateral_force
        
        mujoco.mj_step(model, data)
        step_count += 1
        
        # Record data every ~0.1 seconds
        if step_count % 33 == 0:
            curr_x = data.qpos[0]
            curr_y = data.qpos[1]
            curr_rot = data.qpos[2]
            distance = curr_x - start_x
            
            # Foot heights
            left_z = data.xpos[foot_left_id][2] if foot_left_id else 0
            right_z = data.xpos[foot_right_id][2] if foot_right_id else 0
            
            positions.append((data.time, curr_x, curr_y, curr_rot, distance))
            foot_heights['time'].append(data.time)
            foot_heights['left'].append(left_z)
            foot_heights['right'].append(right_z)
            
            status_left = "ON SAND" if 0.442 <= left_z <= 0.460 else f"OFF ({left_z:.4f}m)"
            status_right = "ON SAND" if 0.442 <= right_z <= 0.460 else f"OFF ({right_z:.4f}m)"
            
            print(f"T:{data.time:6.2f}s | X:{curr_x:+7.4f}m | Dist:{distance:+7.4f}m | Rot:{curr_rot:+7.4f}rad | L:{status_left:15s} | R:{status_right:15s} | F:{forward_force:+10.1f}N")
    
    print()
    print("=" * 100)
    print("SUMMARY:")
    print("=" * 100)
    
    final_x = data.qpos[0]
    total_dist = final_x - start_x
    
    print(f"Initial X: {start_x:.4f}m")
    print(f"Final X: {final_x:.4f}m")
    print(f"Total distance: {total_dist:.4f}m ({total_dist*100:.2f}cm)")
    print(f"Speed: {total_dist/30.0:.4f} m/s ({total_dist*100/30:.2f} cm/s)")
    print()
    
    # Foot heights summary
    left_avg = np.mean(foot_heights['left'])
    right_avg = np.mean(foot_heights['right'])
    left_min = np.min(foot_heights['left'])
    right_min = np.min(foot_heights['right'])
    left_max = np.max(foot_heights['left'])
    right_max = np.max(foot_heights['right'])
    
    print(f"Left foot Z: min={left_min:.4f}m, avg={left_avg:.4f}m, max={left_max:.4f}m")
    print(f"Right foot Z: min={right_min:.4f}m, avg={right_avg:.4f}m, max={right_max:.4f}m")
    print(f"Sand Z range: 0.442-0.450m (contact zone)")
    print()
    
    if total_dist < 0.01:
        print("❌ PROBLEM: Robot not moving forward (< 1cm in 30 seconds)")
    else:
        print(f"✓ Robot moving forward at {total_dist*100/30:.2f} cm/s")
    
    if left_min < 0.440 or right_min < 0.440:
        print("❌ PROBLEM: One or both feet breaking through sand")
    else:
        print("✓ Both feet staying on sand surface")

if __name__ == "__main__":
    main()
