#!/usr/bin/env python3
"""
Terminal-based visualization of robot walking on sand
Shows real-time position, velocity, and sand contact info
"""

import numpy as np
import mujoco
from scipy.interpolate import interp1d
import time
import os

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_robot_state(data, model, total_contacts, sand_displaced):
    """Print formatted robot state"""
    hip_id = model.body("hip").id
    base_pos = data.xpos[hip_id]
    base_vel = data.qvel[0:3] if len(data.qvel) >= 3 else np.array([0, 0, 0])
    
    print("\n" + "="*70)
    print("ROBOT WALKING ON SAND - REAL-TIME STATUS")
    print("="*70)
    print(f"\nTime: {data.time:7.2f}s")
    print(f"\nBase Position:")
    print(f"  X (Forward):  {base_pos[0]:+8.4f}m  |  Velocity: {base_vel[0]:+.4f} m/s")
    print(f"  Y (Lateral):  {base_pos[1]:+8.4f}m  |  Velocity: {base_vel[1]:+.4f} m/s")
    print(f"  Z (Height):   {base_pos[2]:8.4f}m   |  Velocity: {base_vel[2]:+.4f} m/s")
    
    print(f"\nJoint Angles (rad):")
    print(f"  Left Hip:     {data.qpos[3]:+7.4f}  |  Right Hip:   {data.qpos[6]:+7.4f}")
    print(f"  Left Knee:    {data.qpos[4]:+7.4f}  |  Right Knee:  {data.qpos[7]:+7.4f}")
    print(f"  Left Ankle:   {data.qpos[5]:+7.4f}  |  Right Ankle: {data.qpos[8]:+7.4f}")
    
    print(f"\nContact Information:")
    print(f"  Total foot-sand contacts:  {total_contacts:,}")
    print(f"  Sand particles displaced:  {sand_displaced} / 1000")
    
    print("\n" + "="*70)

def main():
    # Load model
    model = mujoco.MjModel.from_xml_path("legged_robot_sand_low_damping.xml")
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
    
    # Parameters
    Kp, Kd = 800.0, 80.0
    gait_period = 3.0
    total_sim_time = 20.0  # 20 seconds
    
    # Get IDs
    foot1_id = model.body("foot_1").id
    foot2_id = model.body("foot_2").id
    
    # Initial sand positions for displacement tracking
    sand_pos_init = data.xpos[1:1001].copy()
    
    print("\nStarting walking simulation...")
    print("Watch the robot walk forward on sand with friction")
    print(f"Total simulation time: {total_sim_time}s\n")
    
    total_contacts = 0
    last_print_time = 0
    
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
        
        # Base Y control
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
        
        mujoco.mj_step(model, data)
        
        # Print status every 2 seconds
        if data.time - last_print_time >= 2.0:
            sand_pos_current = data.xpos[1:1001]
            sand_disp = np.linalg.norm(sand_pos_current - sand_pos_init, axis=1)
            sand_displaced = np.sum(sand_disp > 0.001)
            
            print_robot_state(data, model, total_contacts, sand_displaced)
            last_print_time = data.time
    
    # Final analysis
    sand_pos_final = data.xpos[1:1001]
    sand_disp = np.linalg.norm(sand_pos_final - sand_pos_init, axis=1)
    sand_displaced = np.sum(sand_disp > 0.001)
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Robot base X position:      {data.xpos[model.body('hip').id][0]:+.4f}m")
    print(f"  Total time simulated:       {data.time:.2f}s")
    print(f"  Total foot-sand contacts:   {total_contacts:,}")
    print(f"  Sand particles displaced:   {sand_displaced} / 1000")
    print(f"  Max sand displacement:      {np.max(sand_disp)*1000:.2f}mm")
    print(f"  Avg sand displacement:      {np.mean(sand_disp)*1000:.2f}mm")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
