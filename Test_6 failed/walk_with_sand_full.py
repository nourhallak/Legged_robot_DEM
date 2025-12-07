#!/usr/bin/env python3
"""
Full walking simulation with sand friction
"""

import numpy as np
import mujoco
from mujoco import viewer
from scipy.interpolate import interp1d

def main():
    """Walk on sand with friction simulation."""
    
    print("\n" + "="*80)
    print("BIPEDAL ROBOT WALKING ON SAND WITH FRICTION")
    print("="*80 + "\n")
    
    # Load model with sand
    model = mujoco.MjModel.from_xml_path("legged_robot_sand.xml")
    data = mujoco.MjData(model)
    print("[+] Loaded robot model with sand")
    
    # Load IK solutions
    try:
        ik_times = np.load("ik_times.npy")
        ik_left_hip = np.load("ik_left_hip.npy")
        ik_left_knee = np.load("ik_left_knee.npy")
        ik_left_ankle = np.load("ik_left_ankle.npy")
        ik_right_hip = np.load("ik_right_hip.npy")
        ik_right_knee = np.load("ik_right_knee.npy")
        ik_right_ankle = np.load("ik_right_ankle.npy")
        
        print(f"[+] Loaded IK solutions: {len(ik_times)} points")
    except FileNotFoundError as e:
        print(f"[-] ERROR: Missing IK file: {e}")
        return
    
    # Create interpolations
    print("[+] Creating spline interpolations...")
    interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
    interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
    interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
    interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
    interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
    interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')
    
    # Control parameters
    Kp = 300.0
    Kd = 30.0
    base_Kp = 2500.0
    base_Kd = 150.0
    
    # Sand interaction parameters
    sand_friction_coeff = 0.5
    sand_internal_friction = 0.3
    foot_sink_depth = 0.004
    sand_drag_coeff = 0.6
    
    # Gait parameters
    gait_period = 50.0  # 50 second steps (slow motion for observation)
    num_steps = 3
    total_sim_time = gait_period * num_steps
    
    print(f"\n[+] Walking parameters:")
    print(f"    - Gait cycle: {gait_period}s")
    print(f"    - Number of steps: {num_steps}")
    print(f"    - Total time: {total_sim_time}s")
    print(f"    - Sand friction: {sand_friction_coeff}")
    print(f"[+] Starting headless simulation (no viewer)...\n")
    
    # Get foot body IDs
    foot1_id = model.body("foot_1").id
    foot2_id = model.body("foot_2").id
    
    # Storage for analysis
    base_x_history = []
    contact_history = []
    
    # Main simulation loop (headless)
    step_count = 0
    total_contacts = 0
    
    while data.time < total_sim_time:
        # Update progress every 10 seconds
        if int(data.time) % 10 == 0 and int(data.time) != step_count:
            step_count = int(data.time)
            print(f"[T={data.time:6.1f}s] Base X={data.xpos[0][0]:.4f}m, Contacts={total_contacts}")
        
        # Time within cycle
        t_in_cycle = data.time % gait_period
        
        # Get desired joint angles
        des_left_hip = interp_left_hip(t_in_cycle)
        des_left_knee = interp_left_knee(t_in_cycle)
        des_left_ankle = interp_left_ankle(t_in_cycle)
        des_right_hip = interp_right_hip(t_in_cycle)
        des_right_knee = interp_right_knee(t_in_cycle)
        des_right_ankle = interp_right_ankle(t_in_cycle)
        
        # Joint control (PD) - only 6 motors for leg joints
        data.ctrl[0] = Kp * (des_left_hip - data.qpos[3]) - Kd * data.qvel[3]
        data.ctrl[1] = Kp * (des_left_knee - data.qpos[4]) - Kd * data.qvel[4]
        data.ctrl[2] = Kp * (des_left_ankle - data.qpos[5]) - Kd * data.qvel[5]
        data.ctrl[3] = Kp * (des_right_hip - data.qpos[6]) - Kd * data.qvel[6]
        data.ctrl[4] = Kp * (des_right_knee - data.qpos[7]) - Kd * data.qvel[7]
        data.ctrl[5] = Kp * (des_right_ankle - data.qpos[8]) - Kd * data.qvel[8]
        
        # Base position control via passive joint damping (base joints have no motors)
        # Apply forces to keep base centered
        target_y = 0.0
        data.xfrc_applied[0, 1] = base_Kp * (target_y - data.qpos[1]) - base_Kd * data.qvel[1]
        
        # Apply sand friction
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # Check if foot-sand contact
            is_foot = geom1 in [1004, 1007] or geom2 in [1004, 1007]
            is_sand = (1 <= geom1 <= 1000) or (1 <= geom2 <= 1000)
            
            if is_foot and is_sand:
                total_contacts += 1
                
                # Get foot and sand body IDs
                foot_geom = geom1 if geom1 in [1004, 1007] else geom2
                sand_geom = geom2 if geom1 in [1004, 1007] else geom1
                
                # Map geom to body - sand geoms are 1-1000, so body is geom itself
                sand_body_id = sand_geom
                foot_body_id = foot1_id if foot_geom == 1004 else foot2_id
                
                # Contact normal force (estimate from contact)
                normal_force = np.linalg.norm(contact.frame[0:3])
                
                if normal_force > 0.01:  # Minimum contact force
                    # Get body velocities from cvel
                    foot_cvel = data.cvel[foot_body_id]  # [lin_vel, ang_vel]
                    sand_cvel = data.cvel[sand_body_id]
                    
                    foot_vel = foot_cvel[0:3]
                    sand_vel = sand_cvel[0:3]
                    rel_vel = foot_vel - sand_vel
                    rel_vel_mag = np.linalg.norm(rel_vel)
                    
                    # Friction force
                    if rel_vel_mag > 1e-6:
                        friction_force = -sand_friction_coeff * normal_force * (rel_vel / rel_vel_mag)
                        data.xfrc_applied[sand_body_id, 0:3] += friction_force * 0.01
                    
                    # Backward push (sand behind foot in -X direction)
                    foot_x = data.xpos[foot_body_id][0]
                    sand_x = data.xpos[sand_body_id][0]
                    if sand_x < foot_x:  # Sand is behind foot
                        push_force_x = -2.0 * 0.005
                        data.xfrc_applied[sand_body_id, 0] += push_force_x
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Record history
        base_x_history.append(data.xpos[0][0])
        contact_history.append(total_contacts)
    
    print(f"\n[+] Simulation complete!")
    print(f"[+] Total foot-sand contact instances: {total_contacts}")
    print(f"[+] Final base X: {data.xpos[0][0]:.4f}m")

if __name__ == "__main__":
    main()
