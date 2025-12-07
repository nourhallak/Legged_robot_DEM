#!/usr/bin/env python3
"""
Visualize robot walking on sand with MuJoCo viewer
Real-time 3D visualization of biped robot and sand particles
"""

import numpy as np
import mujoco
from mujoco import viewer
from scipy.interpolate import interp1d
import time

def main():
    """Visualize robot walking on sand."""
    
    print("\n" + "="*80)
    print("BIPEDAL ROBOT WALKING ON SAND - INTERACTIVE VISUALIZATION")
    print("="*80)
    print("\nLoading model and IK trajectories...")
    
    # Load model
    model = mujoco.MjModel.from_xml_path("legged_robot_sand.xml")
    data = mujoco.MjData(model)
    
    # Load IK solutions
    ik_times = np.load("ik_times.npy")
    ik_left_hip = np.load("ik_left_hip.npy")
    ik_left_knee = np.load("ik_left_knee.npy")
    ik_left_ankle = np.load("ik_left_ankle.npy")
    ik_right_hip = np.load("ik_right_hip.npy")
    ik_right_knee = np.load("ik_right_knee.npy")
    ik_right_ankle = np.load("ik_right_ankle.npy")
    
    # Create interpolations
    interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
    interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
    interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
    interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
    interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
    interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')
    
    # Control parameters
    Kp = 600.0
    Kd = 60.0
    
    # Gait parameters
    gait_period = 5.0
    num_steps = 8  # Extended to 8 steps for longer viewing
    total_sim_time = gait_period * num_steps
    
    # Get body IDs
    foot1_id = model.body("foot_1").id
    foot2_id = model.body("foot_2").id
    hip_id = model.body("hip").id
    
    print(f"[+] Model loaded: {model.nbody} bodies, {model.ngeom} geometries")
    print(f"[+] IK trajectory: {len(ik_times)} points ({ik_times[0]:.1f}s to {ik_times[-1]:.1f}s)")
    print(f"[+] Simulation: {num_steps} steps x {gait_period}s = {total_sim_time}s")
    print(f"[+] Launching viewer...\n")
    
    # Launch viewer
    with viewer.launch(model) as v:
        total_contacts = 0
        last_print = 0
        
        while data.time < total_sim_time:
            # Print progress every 2 seconds
            if data.time - last_print >= 2.0:
                progress = (data.time / total_sim_time) * 100
                print(f"[{progress:5.1f}%] T={data.time:6.2f}s | Base: X={data.xpos[hip_id][0]:+.4f}m, " +
                      f"Y={data.xpos[hip_id][1]:+.4f}m | Contacts: {total_contacts}")
                last_print = data.time
            
            # Time in cycle
            t_cycle = data.time % gait_period
            
            # Get desired joint angles from IK
            des_left_hip = interp_left_hip(t_cycle)
            des_left_knee = interp_left_knee(t_cycle)
            des_left_ankle = interp_left_ankle(t_cycle)
            des_right_hip = interp_right_hip(t_cycle)
            des_right_knee = interp_right_knee(t_cycle)
            des_right_ankle = interp_right_ankle(t_cycle)
            
            # Joint control (PD)
            data.ctrl[0] = Kp * (des_left_hip - data.qpos[3]) - Kd * data.qvel[3]
            data.ctrl[1] = Kp * (des_left_knee - data.qpos[4]) - Kd * data.qvel[4]
            data.ctrl[2] = Kp * (des_left_ankle - data.qpos[5]) - Kd * data.qvel[5]
            data.ctrl[3] = Kp * (des_right_hip - data.qpos[6]) - Kd * data.qvel[6]
            data.ctrl[4] = Kp * (des_right_knee - data.qpos[7]) - Kd * data.qvel[7]
            data.ctrl[5] = Kp * (des_right_ankle - data.qpos[8]) - Kd * data.qvel[8]
            
            # Base lateral control
            data.xfrc_applied[0, 1] = 2500.0 * (0.0 - data.qpos[1]) - 150.0 * data.qvel[1]
            
            # Sand friction forces
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
                        
                        # Friction
                        if rel_vel_mag > 1e-6:
                            friction = -0.2 * normal_force * (rel_vel / rel_vel_mag)
                            data.xfrc_applied[sand_body_id, 0:3] += friction * 0.01
                        
                        # Backward push
                        foot_x = data.xpos[foot_body_id][0]
                        sand_x = data.xpos[sand_body_id][0]
                        if sand_x < foot_x:
                            data.xfrc_applied[sand_body_id, 0] += -2.0 * 0.005
            
            # Step simulation
            mujoco.mj_step(model, data)
            v.sync()
    
    print(f"\n[+] Simulation complete!")
    print(f"[+] Total foot-sand contacts: {total_contacts}")
    print(f"[+] Final base position: ({data.xpos[hip_id][0]:+.4f}, {data.xpos[hip_id][1]:+.4f})m")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
