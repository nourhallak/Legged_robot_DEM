#!/usr/bin/env python3
"""
Advanced walking simulation with forward momentum from leg motion
Uses higher control gains and aggressive forward swing
"""

import numpy as np
import mujoco
from scipy.interpolate import interp1d
import time

def main():
    # Load model with sand shifted to positive X (robot walks forward INTO sand)
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
    
    # Interpolations
    interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
    interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
    interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
    interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
    interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
    interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')
    
    # Control parameters - HIGH GAINS for aggressive walking
    Kp, Kd = 1000.0, 100.0  # Increased from 800/80
    gait_period = 3.0
    total_sim_time = 30.0
    
    # Get body IDs
    foot1_id = model.body("foot_1").id
    foot2_id = model.body("foot_2").id
    hip_id = model.body("hip").id
    
    # Initial sand positions
    sand_pos_init = data.xpos[1:1001].copy()
    
    total_contacts = 0
    max_x_reached = 0
    
    print("\n" + "="*70)
    print("FORWARD WALKING SIMULATION - High Gain Control")
    print("="*70)
    print(f"Control: Kp={Kp}, Kd={Kd}")
    print(f"Gait period: {gait_period}s")
    print(f"Total time: {total_sim_time}s\n")
    
    last_print = 0
    has_sand_contact = False  # Track if touching sand
    
    while data.time < total_sim_time:
        # Time in gait cycle
        t_cycle = data.time % gait_period
        
        # JOINT CONTROL - High gains for powerful leg motion
        data.ctrl[0] = Kp * (interp_left_hip(t_cycle) - data.qpos[3]) - Kd * data.qvel[3]
        data.ctrl[1] = Kp * (interp_left_knee(t_cycle) - data.qpos[4]) - Kd * data.qvel[4]
        data.ctrl[2] = Kp * (interp_left_ankle(t_cycle) - data.qpos[5]) - Kd * data.qvel[5]
        data.ctrl[3] = Kp * (interp_right_hip(t_cycle) - data.qpos[6]) - Kd * data.qvel[6]
        data.ctrl[4] = Kp * (interp_right_knee(t_cycle) - data.qpos[7]) - Kd * data.qvel[7]
        data.ctrl[5] = Kp * (interp_right_ankle(t_cycle) - data.qpos[8]) - Kd * data.qvel[8]
        
        # BASE CONTROL - Keep Y lateral position, no forward force needed
        # Y control: keep at y=0 for lateral stability
        data.xfrc_applied[0, 1] = 2500.0 * (0.0 - data.qpos[1]) - 150.0 * data.qvel[1]
        # No forward force - robot naturally walks forward into sand with shifted position
        
        # SAND FRICTION - Apply friction forces
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
        
        # Track forward progress
        base_x = data.xpos[hip_id][0]
        if base_x < max_x_reached:
            max_x_reached = base_x
        
        # Print status every 5 seconds
        if data.time - last_print >= 5.0:
            sand_pos = data.xpos[1:1001]
            sand_disp = np.linalg.norm(sand_pos - sand_pos_init, axis=1)
            sand_moved = np.sum(sand_disp > 0.001)
            
            print(f"Time: {data.time:6.1f}s | Base X: {base_x:+.4f}m | Contacts: {total_contacts:,d} | Sand: {sand_moved}/1000")
            last_print = data.time
    
    # Final analysis
    sand_pos_final = data.xpos[1:1001]
    sand_disp = np.linalg.norm(sand_pos_final - sand_pos_init, axis=1)
    sand_moved = np.sum(sand_disp > 0.001)
    base_x_final = data.xpos[hip_id][0]
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Final robot base X:         {base_x_final:+.4f}m")
    print(f"Maximum forward progress:   {max_x_reached:+.4f}m")
    print(f"Total distance traveled:    {abs(max_x_reached - base_x_final):.4f}m")
    print(f"Total foot-sand contacts:   {total_contacts:,}")
    print(f"Sand particles moved:       {sand_moved}/1000")
    print(f"Average displacement:       {np.mean(sand_disp)*1000:.2f}mm")
    print(f"Maximum displacement:       {np.max(sand_disp)*1000:.2f}mm")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
