#!/usr/bin/env python3
"""
SLOW MOTION WALKING DEMO
Simplified robot walking on sand - slowed down 10x for visibility
"""

import numpy as np
import mujoco
from mujoco import viewer
from scipy.interpolate import interp1d

def main():
    """Simulate robot walking on sand at slow speed."""
    
    print("\n" + "="*100)
    print("SLOW MOTION WALKING DEMO - 10X SLOWER FOR VISIBILITY")
    print("="*100 + "\n")
    
    # Load model with sand
    model = mujoco.MjModel.from_xml_path("legged_robot_sand_shifted.xml")
    data = mujoco.MjData(model)
    print("[+] Loaded robot model with sand")
    
    # Reset to initial state - robot starts at beginning of sand
    mujoco.mj_resetData(model, data)
    print("[+] Reset robot to initial state")
    
    # Load IK solutions (joint angles)
    ik_times = np.load("ik_times.npy")
    ik_left_hip = np.load("ik_left_hip.npy")
    ik_left_knee = np.load("ik_left_knee.npy")
    ik_left_ankle = np.load("ik_left_ankle.npy")
    ik_right_hip = np.load("ik_right_hip.npy")
    ik_right_knee = np.load("ik_right_knee.npy")
    ik_right_ankle = np.load("ik_right_ankle.npy")
    
    print(f"[+] Loaded IK trajectories: {len(ik_times)} points ({ik_times[-1]:.2f}s)")
    
    # Scale amplitudes for bigger stride
    amplitude_scale = 0.15  # Back to proven stable value
    ik_left_hip *= amplitude_scale
    ik_left_knee *= amplitude_scale
    ik_left_ankle *= amplitude_scale
    ik_right_hip *= amplitude_scale
    ik_right_knee *= amplitude_scale
    ik_right_ankle *= amplitude_scale
    
    # Create interpolation functions for smooth playback
    interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
    interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
    interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
    
    interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
    interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
    interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')
    
    # Control parameters
    Kp = 600.0    # Joint proportional gain
    Kd = 60.0     # Joint derivative gain
    
    # Get body IDs
    hip_id = model.body("hip").id
    try:
        foot_left_id = model.body("foot_1").id
        foot_right_id = model.body("foot_2").id
    except:
        foot_left_id = None
        foot_right_id = None
    
    # Gait parameters
    gait_period = ik_times[-1]  # Full cycle time
    forward_force = 1000.0  # Much stronger forward push for bigger stride
    
    print(f"[+] Control setup: Kp={Kp}, Kd={Kd}")
    print(f"[+] Gait period: {gait_period:.2f}s")
    print(f"[+] Forward push force: {forward_force}N")
    print(f"[+] Amplitude scale: {amplitude_scale*100:.0f}%")
    print(f"[+] Robot starts at X = 0.150m (beginning of sand, Distance = 0.000m)")
    
    print("\n" + "="*100)
    print("STARTING SIMULATION - Watch robot walk slowly on sand")
    print("="*100 + "\n")
    
    step_count = 0
    
    with viewer.launch_passive(model, data) as v:
        v.cam.azimuth = 90
        v.cam.distance = 1.5
        v.cam.elevation = 0
        v.cam.lookat[:] = [0.3, 0, 0.5]
        
        # Slow motion playback - use 10x slower timestep
        model.opt.timestep = 0.001  # 1ms per step
        
        while v.is_running() and data.time < 120:  # Run for 2 minutes
            
            # Get current position in gait cycle (with slow motion)
            # Divide time by 10 to slow things down
            t_gait = (data.time / 10.0) % gait_period
            
            # Get desired joint angles from interpolation
            q_left_hip_des = interp_left_hip(t_gait)
            q_left_knee_des = interp_left_knee(t_gait)
            q_left_ankle_des = interp_left_ankle(t_gait)
            
            q_right_hip_des = interp_right_hip(t_gait)
            q_right_knee_des = interp_right_knee(t_gait)
            q_right_ankle_des = interp_right_ankle(t_gait)
            
            # Joint control (PD controller on each joint)
            # Left leg: ctrl[0]=hip, ctrl[1]=knee, ctrl[2]=ankle
            data.ctrl[0] = Kp * (q_left_hip_des - data.qpos[3]) - Kd * data.qvel[3]
            data.ctrl[1] = Kp * (q_left_knee_des - data.qpos[4]) - Kd * data.qvel[4]
            data.ctrl[2] = Kp * (q_left_ankle_des - data.qpos[5]) - Kd * data.qvel[5]
            
            # Right leg: ctrl[3]=hip, ctrl[4]=knee, ctrl[5]=ankle
            data.ctrl[3] = Kp * (q_right_hip_des - data.qpos[6]) - Kd * data.qvel[6]
            data.ctrl[4] = Kp * (q_right_knee_des - data.qpos[7]) - Kd * data.qvel[7]
            data.ctrl[5] = Kp * (q_right_ankle_des - data.qpos[8]) - Kd * data.qvel[8]
            
            # Apply forward force to hip body (gently push robot forward)
            data.xfrc_applied[hip_id, 0] = forward_force
            
            # Prevent rotation (yaw control)
            rotation_kp = 30.0
            rotation_kd = 15.0
            torque_z = -rotation_kp * data.qpos[2] - rotation_kd * data.qvel[2]
            data.xfrc_applied[hip_id, 5] = torque_z
            
            # Prevent lateral drift (y control)
            lateral_kp = 80.0
            lateral_kd = 15.0
            force_y = -lateral_kp * (data.qpos[1] - 0.0) - lateral_kd * data.qvel[1]
            data.xfrc_applied[hip_id, 1] = force_y
            
            # Step simulation
            mujoco.mj_step(model, data)
            v.sync()
            
            step_count += 1
            
            # Print every 1000 steps = 1 second of simulation = 0.1s real gait time
            if step_count % 1000 == 0:
                hip_x = data.xpos[hip_id][0]
                hip_y = data.xpos[hip_id][1]
                
                # Check foot contact with sand (Z between 0.442 and 0.46)
                if foot_left_id and foot_right_id:
                    left_z = data.xpos[foot_left_id][2]
                    right_z = data.xpos[foot_right_id][2]
                    left_contact = "ON" if 0.442 <= left_z <= 0.46 else "OFF"
                    right_contact = "ON" if 0.442 <= right_z <= 0.46 else "OFF"
                else:
                    left_z = right_z = 0
                    left_contact = right_contact = "?"
                
                distance = hip_x - 0.150
                gait_progress = (t_gait / gait_period) * 100
                
                print(f"T: {data.time:7.2f}s | Gait: {gait_progress:5.1f}% | X: {hip_x:.4f}m | Dist: {distance:+.4f}m | " +
                      f"L-foot: Z={left_z:.4f}m {left_contact} | R-foot: Z={right_z:.4f}m {right_contact}")

if __name__ == "__main__":
    main()
