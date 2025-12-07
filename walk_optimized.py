#!/usr/bin/env python3
"""
Optimized walking simulation on sand - faster gait with stronger actuators
"""

import numpy as np
import mujoco
from scipy.interpolate import interp1d

def main():
    """Walk on sand with optimized parameters."""
    
    print("\n" + "="*80)
    print("BIPEDAL ROBOT WALKING ON SAND - OPTIMIZED (FASTER GAIT)")
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
    
    # OPTIMIZED Control parameters - STRONGER ACTUATION
    Kp = 600.0      # DOUBLED from 300
    Kd = 60.0       # DOUBLED from 30
    base_Kp = 2500.0
    base_Kd = 150.0
    
    # Sand interaction parameters - REDUCED friction for forward motion
    sand_friction_coeff = 0.2    # REDUCED from 0.5 (less grip = more sliding)
    sand_internal_friction = 0.3
    foot_sink_depth = 0.004
    sand_drag_coeff = 0.2        # REDUCED from 0.6 (less drag)
    
    # OPTIMIZED Gait parameters - FASTER WALKING
    gait_period = 5.0   # REDUCED from 50s to 5s (10x faster!)
    num_steps = 6       # 6 steps for longer distance
    total_sim_time = gait_period * num_steps
    
    print(f"\n[+] OPTIMIZED PARAMETERS:")
    print(f"    - Gait cycle: {gait_period}s (10x faster)")
    print(f"    - Actuator Kp: {Kp} (2x stronger)")
    print(f"    - Actuator Kd: {Kd} (2x stronger)")
    print(f"    - Number of steps: {num_steps}")
    print(f"    - Total time: {total_sim_time}s")
    print(f"\n[+] Starting headless simulation...\n")
    
    # Get foot body IDs
    foot1_id = model.body("foot_1").id
    foot2_id = model.body("foot_2").id
    
    # Storage for analysis
    base_x_history = []
    base_y_history = []
    time_history = []
    contact_count_history = []
    
    # Sand displacement tracking
    sand_positions_initial = data.xpos[1:1001].copy()
    
    # Main simulation loop
    step_count = 0
    total_contacts = 0
    last_print_time = 0
    
    while data.time < total_sim_time:
        # Print progress every 1 second
        if data.time - last_print_time >= 1.0:
            print(f"[T={data.time:6.2f}s] Base X={data.xpos[0][0]:+.4f}m, Y={data.xpos[0][1]:+.4f}m, " +
                  f"Contacts/s={total_contacts - (last_print_time * 100000):.0f}")
            last_print_time = data.time
        
        # Time within cycle
        t_in_cycle = data.time % gait_period
        
        # Get desired joint angles
        des_left_hip = interp_left_hip(t_in_cycle)
        des_left_knee = interp_left_knee(t_in_cycle)
        des_left_ankle = interp_left_ankle(t_in_cycle)
        des_right_hip = interp_right_hip(t_in_cycle)
        des_right_knee = interp_right_knee(t_in_cycle)
        des_right_ankle = interp_right_ankle(t_in_cycle)
        
        # Joint control (PD) - with stronger gains
        data.ctrl[0] = Kp * (des_left_hip - data.qpos[3]) - Kd * data.qvel[3]
        data.ctrl[1] = Kp * (des_left_knee - data.qpos[4]) - Kd * data.qvel[4]
        data.ctrl[2] = Kp * (des_left_ankle - data.qpos[5]) - Kd * data.qvel[5]
        data.ctrl[3] = Kp * (des_right_hip - data.qpos[6]) - Kd * data.qvel[6]
        data.ctrl[4] = Kp * (des_right_knee - data.qpos[7]) - Kd * data.qvel[7]
        data.ctrl[5] = Kp * (des_right_ankle - data.qpos[8]) - Kd * data.qvel[8]
        
        # Base position control via applied forces
        target_y = 0.0
        data.xfrc_applied[0, 1] = base_Kp * (target_y - data.qpos[1]) - base_Kd * data.qvel[1]
        
        # DEBUG: Add small forward force to base to test movement
        # Remove this once we see the base can move
        data.xfrc_applied[0, 0] += 0.1  # Small forward push
        
        # Apply sand friction forces
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
                
                sand_body_id = sand_geom
                foot_body_id = foot1_id if foot_geom == 1004 else foot2_id
                
                # Contact normal force
                normal_force = np.linalg.norm(contact.frame[0:3])
                
                if normal_force > 0.01:
                    # Get body velocities
                    foot_cvel = data.cvel[foot_body_id]
                    sand_cvel = data.cvel[sand_body_id]
                    
                    foot_vel = foot_cvel[0:3]
                    sand_vel = sand_cvel[0:3]
                    rel_vel = foot_vel - sand_vel
                    rel_vel_mag = np.linalg.norm(rel_vel)
                    
                    # Friction force - increased coefficient for more traction
                    if rel_vel_mag > 1e-6:
                        friction_force = -sand_friction_coeff * normal_force * (rel_vel / rel_vel_mag)
                        data.xfrc_applied[sand_body_id, 0:3] += friction_force * 0.01
                    
                    # Backward push - sand behind foot is pushed back
                    foot_x = data.xpos[foot_body_id][0]
                    sand_x = data.xpos[sand_body_id][0]
                    if sand_x < foot_x:
                        push_force_x = -2.0 * 0.005
                        data.xfrc_applied[sand_body_id, 0] += push_force_x
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Record history
        base_x_history.append(data.xpos[0][0])
        base_y_history.append(data.xpos[0][1])
        time_history.append(data.time)
        contact_count_history.append(total_contacts)
    
    # Analysis
    print(f"\n" + "="*80)
    print("SIMULATION RESULTS")
    print("="*80)
    print(f"\n[+] Total simulation time: {data.time:.2f}s")
    print(f"[+] Total foot-sand contacts: {total_contacts}")
    print(f"[+] Average contacts per step: {total_contacts / num_steps:.0f}")
    
    print(f"\n[+] Robot BASE POSITION:")
    print(f"    - Initial X: {base_x_history[0]:+.4f}m")
    print(f"    - Final X: {base_x_history[-1]:+.4f}m")
    print(f"    - Distance traveled: {base_x_history[-1] - base_x_history[0]:+.4f}m")
    print(f"    - Final Y (lateral): {base_y_history[-1]:+.4f}m")
    
    print(f"\n[+] SAND PARTICLE DISPLACEMENT:")
    sand_positions_final = data.xpos[1:1001]
    sand_displacements = np.linalg.norm(sand_positions_final - sand_positions_initial, axis=1)
    
    moved_particles = np.sum(sand_displacements > 0.001)
    avg_displacement = np.mean(sand_displacements)
    max_displacement = np.max(sand_displacements)
    
    print(f"    - Particles moved (>1mm): {moved_particles} / 1000")
    print(f"    - Average displacement: {avg_displacement:.6f}m")
    print(f"    - Maximum displacement: {max_displacement:.6f}m")
    
    if moved_particles > 0:
        print("SUCCESS: Sand interaction detected!")
        print(f"  {(moved_particles/1000)*100:.1f}% of sand particles were displaced")
    else:
        print("No sand movement detected - check friction parameters")
    
    print(f"\n[+] VELOCITY ANALYSIS:")
    final_vel_x = data.qvel[0] if len(data.qvel) > 0 else 0
    final_vel_y = data.qvel[1] if len(data.qvel) > 1 else 0
    print(f"    - Final X velocity: {final_vel_x:+.4f} m/s")
    print(f"    - Final Y velocity: {final_vel_y:+.4f} m/s")
    
    return {
        'time': data.time,
        'distance': base_x_history[-1] - base_x_history[0],
        'contacts': total_contacts,
        'particles_moved': moved_particles,
        'max_sand_displacement': max_displacement
    }

if __name__ == "__main__":
    results = main()
