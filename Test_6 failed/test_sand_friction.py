#!/usr/bin/env python3
"""
Test sand friction and sinking without viewer (headless mode)
"""

import numpy as np
import mujoco
from scipy.interpolate import interp1d

def main():
    """Test robot walking on sand with friction."""
    
    print("\n" + "="*80)
    print("BIPEDAL ROBOT WALKING ON SAND - FRICTION & SINKING TEST (HEADLESS)")
    print("="*80 + "\n")
    
    # Load model with sand
    model = mujoco.MjModel.from_xml_path("legged_robot_sand.xml")
    data = mujoco.MjData(model)
    print("[+] Loaded robot model with sand: legged_robot_sand.xml")
    
    # Load IK solutions (joint angles)
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
        print(f"[-] ERROR: Missing file: {e}")
        return
    
    # Create interpolation functions for smooth playback
    print("[+] Creating cubic spline interpolations...")
    
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
    sand_friction_coeff = 0.5  # Coefficient of friction between foot and sand
    sand_internal_friction = 0.3  # Friction between sand particles
    foot_sink_depth = 0.004  # Maximum sinking depth (4mm)
    sand_drag_coeff = 0.6  # Resistance to sand motion
    
    # Gait cycle and walking
    gait_period = 50.0
    num_steps = 5  # Just 5 steps for testing
    total_sim_time = gait_period * num_steps
    
    print(f"[+] Walking parameters:")
    print(f"    - Gait cycle: {gait_period}s")
    print(f"    - Number of steps: {num_steps}")
    print(f"    - Total time: {total_sim_time}s")
    print(f"    - Sand friction coeff: {sand_friction_coeff}")
    print(f"    - Foot sink depth: {foot_sink_depth*1000:.1f}mm")
    print(f"    - Sand drag coeff: {sand_drag_coeff}")
    print(f"\n[+] Starting headless simulation (no viewer)...\n")
    
    # Storage for analysis
    base_x_history = []
    sand_movement = []
    initial_sand_pos = data.xpos[1:1001].copy()  # Store initial positions of first 1000 particles
    
    # Main simulation loop
    count = 0
    contact_count = 0
    
    while data.time < total_sim_time:
        # Calculate time within current cycle
        t_in_cycle = data.time % gait_period
        
        # Get desired joint angles
        des_left_hip = interp_left_hip(t_in_cycle)
        des_left_knee = interp_left_knee(t_in_cycle)
        des_left_ankle = interp_left_ankle(t_in_cycle)
        
        des_right_hip = interp_right_hip(t_in_cycle)
        des_right_knee = interp_right_knee(t_in_cycle)
        des_right_ankle = interp_right_ankle(t_in_cycle)
        
        # Joint control
        data.ctrl[0] = Kp * (des_left_hip - data.qpos[3]) - Kd * data.qvel[3]
        data.ctrl[1] = Kp * (des_left_knee - data.qpos[4]) - Kd * data.qvel[4]
        data.ctrl[2] = Kp * (des_left_ankle - data.qpos[5]) - Kd * data.qvel[5]
        
        data.ctrl[3] = Kp * (des_right_hip - data.qpos[6]) - Kd * data.qvel[6]
        data.ctrl[4] = Kp * (des_right_knee - data.qpos[7]) - Kd * data.qvel[7]
        data.ctrl[5] = Kp * (des_right_ankle - data.qpos[8]) - Kd * data.qvel[8]
        
        # Base control
        target_base_x = data.time * 0.0018
        target_base_y = 0.0
        
        forward_force = base_Kp * (target_base_x - data.qpos[0]) - base_Kd * data.qvel[0]
        data.xfrc_applied[1001, 0] = forward_force
        
        lateral_force = base_Kp * (target_base_y - data.qpos[1]) - base_Kd * data.qvel[1]
        data.xfrc_applied[1001, 1] = lateral_force
        
        rotation_damping = 5.0
        data.xfrc_applied[1001, 5] = -rotation_damping * data.qvel[2]
        
        # ============================================================================
        # SAND INTERACTION WITH FRICTION AND SINKING
        # ============================================================================
        contacted_sand = {}
        
        # Get foot positions
        left_foot_pos = data.xpos[4].copy()
        right_foot_pos = data.xpos[9].copy()
        
        # Process all contacts
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1_body = model.geom_bodyid[contact.geom1]
            geom2_body = model.geom_bodyid[contact.geom2]
            
            foot_id = None
            sand_id = None
            
            if geom1_body in [4, 9] and 1 <= geom2_body <= 1000:
                foot_id = geom1_body
                sand_id = geom2_body
            elif geom2_body in [4, 9] and 1 <= geom1_body <= 1000:
                foot_id = geom2_body
                sand_id = geom1_body
            
            if sand_id is not None:
                contacted_sand[sand_id] = foot_id
                contact_count += 1
                
                # Get positions and velocities
                sand_pos = data.xpos[sand_id].copy()
                foot_pos = data.xpos[foot_id].copy()
                foot_vel = data.cvel[foot_id][:3]
                sand_vel = data.cvel[sand_id][:3]
                
                # Relative velocity
                rel_vel = foot_vel - sand_vel
                rel_vel_mag = np.linalg.norm(rel_vel)
                
                # ====== FRICTION-BASED FORCE ======
                if rel_vel_mag > 1e-6:
                    dist_to_foot = np.linalg.norm(sand_pos - foot_pos)
                    normal_force = max(1.0, 10.0 - dist_to_foot * 100)
                    friction_force = -sand_friction_coeff * normal_force * (rel_vel / rel_vel_mag)
                else:
                    friction_force = np.array([0.0, 0.0, 0.0])
                
                # ====== FOOT SINKING EFFECT ======
                push_down_force = 0.0
                if abs(sand_pos[0] - foot_pos[0]) < 0.01 and abs(sand_pos[1] - foot_pos[1]) < 0.01:
                    sand_depth = foot_pos[2] - sand_pos[2]
                    if 0 < sand_depth < foot_sink_depth:
                        push_down_force = (sand_depth / foot_sink_depth) * 3.0
                
                # ====== BACKWARD PUSH ======
                push_back_force = 0.0
                relative_x = sand_pos[0] - foot_pos[0]
                
                if relative_x < 0.01 and relative_x > -0.04:
                    proximity_factor = 1.0 - (abs(relative_x) / 0.04)
                    push_back_force = -2.0 * proximity_factor
                
                # ====== SAND-SAND COUPLING ======
                drag_force = -sand_drag_coeff * sand_vel
                
                # ====== APPLY ALL FORCES ======
                data.xfrc_applied[sand_id, 0] += friction_force[0] + push_back_force + drag_force[0]
                data.xfrc_applied[sand_id, 1] += friction_force[1] + drag_force[1]
                data.xfrc_applied[sand_id, 2] += friction_force[2] - push_down_force + drag_force[2]
        
        # ====== SAND-SAND FRICTION FOR NON-CONTACTED PARTICLES ======
        for sand_id in range(1, 1001):
            if sand_id not in contacted_sand:
                sand_vel = data.cvel[sand_id][:3]
                internal_damping = 0.5
                data.xfrc_applied[sand_id, 0] -= internal_damping * sand_vel[0]
                data.xfrc_applied[sand_id, 1] -= internal_damping * sand_vel[1]
                data.xfrc_applied[sand_id, 2] -= internal_damping * sand_vel[2]
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Record data
        count += 1
        base_x_history.append(data.qpos[0])
        
        # Progress reporting
        if count % 500 == 0:
            step_in_cycle = t_in_cycle / gait_period
            current_step = int(data.time / gait_period) + 1
            percent = (data.time / total_sim_time) * 100
            avg_contacts = contact_count / max(count, 1)
            print(f"[Step {current_step:2d}/{num_steps}] t={data.time:7.1f}s | {percent:5.1f}% | Base X={data.qpos[0]:7.4f}m | Avg contacts/step: {avg_contacts:.1f}")
    
    # Calculate sand movement
    final_sand_pos = data.xpos[1:1001].copy()
    sand_displacement = np.linalg.norm(final_sand_pos - initial_sand_pos, axis=1)
    avg_sand_disp = np.mean(sand_displacement)
    max_sand_disp = np.max(sand_displacement)
    
    # Final summary
    final_x = data.qpos[0]
    steps_completed = int(data.time / gait_period)
    
    print(f"\n[OK] Simulation completed!")
    print(f"[+] Steps completed: {steps_completed}/{num_steps}")
    print(f"[+] Total simulation time: {data.time:.2f}s")
    print(f"[+] Final base X position: {final_x:.4f}m")
    print(f"[+] Distance traveled: {final_x:.4f}m")
    print(f"[+] Average distance per step: {final_x/max(steps_completed, 1):.4f}m")
    print(f"\n[+] Sand Movement Analysis:")
    print(f"    - Average sand particle displacement: {avg_sand_disp:.6f}m")
    print(f"    - Maximum sand particle displacement: {max_sand_disp:.6f}m")
    print(f"    - Total foot-sand contacts detected: {contact_count}")
    print(f"    - Average contacts per simulation step: {contact_count/max(count, 1):.2f}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
