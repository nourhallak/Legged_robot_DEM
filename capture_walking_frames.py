#!/usr/bin/env python3
"""
Create animation/video of robot walking on sand
Saves frames that can be viewed as a sequence
"""

import numpy as np
import mujoco
from scipy.interpolate import interp1d
import os

def main():
    """Generate walking animation frames."""
    
    print("\n" + "="*80)
    print("BIPEDAL ROBOT WALKING ON SAND - FRAME CAPTURE")
    print("="*80)
    print("\nGenerating animation frames...\n")
    
    # Load model
    model = mujoco.MjModel.from_xml_path("legged_robot_sand.xml")
    data = mujoco.MjData(model)
    
    # Load IK
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
    Kp, Kd = 600.0, 60.0
    gait_period = 5.0
    num_steps = 4
    total_sim_time = gait_period * num_steps
    
    # Get body IDs
    foot1_id = model.body("foot_1").id
    foot2_id = model.body("foot_2").id
    hip_id = model.body("hip").id
    
    # Create frames directory
    frames_dir = "walking_frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    print(f"[+] Saving frames to: {frames_dir}/\n")
    
    # Capture every N steps (every 0.5 seconds = 100 simulation steps at 0.005s timestep)
    capture_interval = 100
    frame_count = 0
    total_contacts = 0
    
    while data.time < total_sim_time:
        # Time in cycle
        t_cycle = data.time % gait_period
        
        # Joint control
        data.ctrl[0] = Kp * (interp_left_hip(t_cycle) - data.qpos[3]) - Kd * data.qvel[3]
        data.ctrl[1] = Kp * (interp_left_knee(t_cycle) - data.qpos[4]) - Kd * data.qvel[4]
        data.ctrl[2] = Kp * (interp_left_ankle(t_cycle) - data.qpos[5]) - Kd * data.qvel[5]
        data.ctrl[3] = Kp * (interp_right_hip(t_cycle) - data.qpos[6]) - Kd * data.qvel[6]
        data.ctrl[4] = Kp * (interp_right_knee(t_cycle) - data.qpos[7]) - Kd * data.qvel[7]
        data.ctrl[5] = Kp * (interp_right_ankle(t_cycle) - data.qpos[8]) - Kd * data.qvel[8]
        
        # Base lateral control
        data.xfrc_applied[0, 1] = 2500.0 * (0.0 - data.qpos[1]) - 150.0 * data.qvel[1]
        
        # Sand friction
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            
            is_foot = geom1 in [1004, 1007] or geom2 in [1004, 1007]
            is_sand = (1 <= geom1 <= 1000) or (1 <= geom2 <= 1000)
            
            if is_foot and is_sand:
                total_contacts += 1
                
                foot_geom = geom1 if geom1 in [1004, 1007] else geom2
                sand_geom = geom2 if geom1 in [1004, 1007] else geom1
                
                sand_body_id = sand_geom
                foot_body_id = foot1_id if foot_geom == 1004 else foot2_id
                
                normal_force = np.linalg.norm(contact.frame[0:3])
                
                if normal_force > 0.01:
                    foot_cvel = data.cvel[foot_body_id]
                    sand_cvel = data.cvel[sand_body_id]
                    rel_vel = foot_cvel[0:3] - sand_cvel[0:3]
                    rel_vel_mag = np.linalg.norm(rel_vel)
                    
                    if rel_vel_mag > 1e-6:
                        friction = -0.2 * normal_force * (rel_vel / rel_vel_mag)
                        data.xfrc_applied[sand_body_id, 0:3] += friction * 0.01
                    
                    foot_x = data.xpos[foot_body_id][0]
                    sand_x = data.xpos[sand_body_id][0]
                    if sand_x < foot_x:
                        data.xfrc_applied[sand_body_id, 0] += -2.0 * 0.005
        
        # Capture frame state
        if int(data.time * 2) % 2 == 0:  # Every 0.5s
            frame_count += 1
            hip_pos = data.xpos[hip_id]
            foot1_pos = data.xpos[foot1_id]
            foot2_pos = data.xpos[foot2_id]
            
            # Save state to file
            state_file = os.path.join(frames_dir, f"frame_{frame_count:04d}.txt")
            with open(state_file, "w") as f:
                f.write(f"Time: {data.time:.2f}s\n")
                f.write(f"Hip: {hip_pos[0]:+.4f}, {hip_pos[1]:+.4f}, {hip_pos[2]:.4f}\n")
                f.write(f"Foot1: {foot1_pos[0]:+.4f}, {foot1_pos[1]:+.4f}, {foot1_pos[2]:.4f}\n")
                f.write(f"Foot2: {foot2_pos[0]:+.4f}, {foot2_pos[1]:+.4f}, {foot2_pos[2]:.4f}\n")
                f.write(f"Contacts: {total_contacts}\n")
            
            if frame_count % 4 == 0:
                print(f"  Frame {frame_count}: T={data.time:.2f}s | Hip: ({hip_pos[0]:+.4f}, {hip_pos[1]:+.4f})m")
        
        mujoco.mj_step(model, data)
    
    print(f"\n[+] Captured {frame_count} frames")
    print(f"[+] Total foot-sand contacts: {total_contacts}")
    print(f"\n[+] Frame files saved in: {frames_dir}/")
    print(f"[+] Example: {frames_dir}/frame_0001.txt")

if __name__ == "__main__":
    main()
