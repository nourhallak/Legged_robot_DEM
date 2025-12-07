#!/usr/bin/env python3
"""
Robot walking simulation on sand with 1000 particles.
Uses pre-computed joint angle trajectories from generate_simple_ik.py
"""

import numpy as np
import mujoco
from mujoco import viewer
from scipy.interpolate import interp1d

def main():
    """Simulate robot walking on sand."""
    
    print("\n" + "="*80)
    print("BIPEDAL ROBOT WALKING ON SAND (1000 PARTICLES)")
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
        print(f"    - Left hip angle range: {ik_left_hip.min():.4f} to {ik_left_hip.max():.4f} rad")
        print(f"    - Left knee angle range: {ik_left_knee.min():.4f} to {ik_left_knee.max():.4f} rad")
        print(f"    - Left ankle angle range: {ik_left_ankle.min():.4f} to {ik_left_ankle.max():.4f} rad")
    except FileNotFoundError as e:
        print(f"[-] ERROR: Missing file: {e}")
        print("[-] Run: python generate_simple_ik.py")
        return
    
    # Create interpolation functions for smooth playback
    print("\n[+] Creating cubic spline interpolations...")
    
    interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
    interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
    interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
    
    interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
    interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
    interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')
    
    print("[+] Interpolation functions created")
    
    # Control parameters
    Kp = 300.0  # Proportional gain for joint control
    Kd = 30.0   # Derivative gain for joint control
    
    # Base motion control
    base_Kp = 2500.0  # Base position gain (stronger lateral control)
    base_Kd = 150.0   # Base velocity damping (increased)
    
    # Sand interaction parameters
    sand_friction_coeff = 0.5  # Coefficient of friction between foot and sand
    sand_internal_friction = 0.3  # Friction between sand particles
    foot_sink_depth = 0.004  # Maximum sinking depth (4mm) for foot on sand
    sand_drag_coeff = 0.6  # Resistance to sand motion (higher = more resistance)
    
    # Gait cycle duration and walking target
    gait_period = 50.0  # One complete walking cycle in seconds
    num_steps = 20  # Number of steps to walk
    total_sim_time = gait_period * num_steps  # 50s per step × 20 steps = 1000s
    
    print(f"\n[+] Walking parameters:")
    print(f"    - Gait cycle duration: {gait_period:.2f}s")
    print(f"    - Number of steps: {num_steps}")
    print(f"    - Total simulation time: {total_sim_time:.0f}s")
    print(f"    - Control gains (joints): Kp={Kp}, Kd={Kd}")
    print(f"    - Control gains (base): Kp={base_Kp}, Kd={base_Kd}")
    print(f"    - Simulation timestep: {model.opt.timestep}s")
    print(f"    - Sand particles: 1000 balls in 3 layers (touching)")
    print(f"    - Sand ball radius: 0.002m (2mm) - reduced for finer sand resolution")
    print(f"    - Gravity: 9.81 m/s² (enabled)")
    
    print(f"\n[+] Starting MuJoCo viewer...")
    print(f"[+] Close the window to stop walking\n")
    
    # Initialize viewer with passive mode
    with viewer.launch_passive(model, data) as v:
        v.cam.azimuth = 90  # Side view
        v.cam.distance = 1.5  # Camera distance from robot
        v.cam.elevation = 0  # View from side
        v.cam.lookat[:] = [0, 0, 0.5]  # Look at robot center
        
        # Main simulation loop
        count = 0
        total_cycles = 0
        
        while v.is_running() and data.time < total_sim_time:
            # Calculate time within current cycle (repeating pattern)
            t_in_cycle = data.time % gait_period
            
            # Get desired joint angles from interpolated trajectory
            des_left_hip = interp_left_hip(t_in_cycle)
            des_left_knee = interp_left_knee(t_in_cycle)
            des_left_ankle = interp_left_ankle(t_in_cycle)
            
            des_right_hip = interp_right_hip(t_in_cycle)
            des_right_knee = interp_right_knee(t_in_cycle)
            des_right_ankle = interp_right_ankle(t_in_cycle)
            
            # Target forward base position - move forward consistently
            # For 20 steps × 50s per step = 1000s, we want to cover ~1.2m (sand length)
            # That's ~0.0012 m/s, but we need faster to overcome leg swing drag
            target_base_x = data.time * 0.0018  # Increased: 1.8mm/s forward velocity
            target_base_y = 0.0  # Keep Y centered (no lateral drift)
            
            # Joint control - PD controller for each actuator
            # Left leg (joints 3, 4, 5 -> controls 0, 1, 2)
            data.ctrl[0] = Kp * (des_left_hip - data.qpos[3]) - Kd * data.qvel[3]
            data.ctrl[1] = Kp * (des_left_knee - data.qpos[4]) - Kd * data.qvel[4]
            data.ctrl[2] = Kp * (des_left_ankle - data.qpos[5]) - Kd * data.qvel[5]
            
            # Right leg (joints 6, 7, 8 -> controls 3, 4, 5)
            data.ctrl[3] = Kp * (des_right_hip - data.qpos[6]) - Kd * data.qvel[6]
            data.ctrl[4] = Kp * (des_right_knee - data.qpos[7]) - Kd * data.qvel[7]
            data.ctrl[5] = Kp * (des_right_ankle - data.qpos[8]) - Kd * data.qvel[8]
            
            # Base forward motion control via external force
            # Body 1001 is the hip (robot), not body 1 which is sand_0_0_0
            forward_force = base_Kp * (target_base_x - data.qpos[0]) - base_Kd * data.qvel[0]
            data.xfrc_applied[1001, 0] = forward_force  # Apply to hip body (body 1001)
            
            # Base lateral (Y) control - prevent drift left or right
            lateral_force = base_Kp * (target_base_y - data.qpos[1]) - base_Kd * data.qvel[1]
            data.xfrc_applied[1001, 1] = lateral_force  # Apply to hip body (body 1001)
            
            # Damp rotation in place: prevent unwanted spinning from asymmetric leg movements
            # Apply counter-torque to stabilize rotation around Z axis
            rotation_damping = 5.0  # Damping coefficient for rotational motion
            data.xfrc_applied[1001, 5] = -rotation_damping * data.qvel[2]  # Torque around Z axis
            
            # ============================================================================
            # SAND INTERACTION SYSTEM - Realistic foot-sand dynamics
            # ============================================================================
            # Track foot positions for friction calculation
            foot_positions = {}  # Store foot positions from previous step
            contacted_sand = {}  # Track which sand particles are being pushed
            
            # Get foot positions (body IDs 4 and 9)
            left_foot_pos = data.xpos[4].copy()
            right_foot_pos = data.xpos[9].copy()
            
            # Process all contacts to handle foot-sand interactions
            for i in range(data.ncon):
                contact = data.contact[i]
                geom1_body = model.geom_bodyid[contact.geom1]
                geom2_body = model.geom_bodyid[contact.geom2]
                
                # Identify foot-sand contact
                foot_id = None
                sand_id = None
                contact_force_z = 0
                
                if geom1_body in [4, 9] and 1 <= geom2_body <= 1000:
                    foot_id = geom1_body
                    sand_id = geom2_body
                elif geom2_body in [4, 9] and 1 <= geom1_body <= 1000:
                    foot_id = geom2_body
                    sand_id = geom1_body
                
                if sand_id is not None:
                    contacted_sand[sand_id] = foot_id
                    
                    # Get contact force (normal force from contact array)
                    # Extract normal force from contact data
                    contact_force = contact.solref if hasattr(contact, 'solref') else 0
                    
                    # Get positions
                    sand_pos = data.xpos[sand_id].copy()
                    foot_pos = data.xpos[foot_id].copy()
                    foot_vel = data.cvel[foot_id][:3]  # Linear velocity only
                    sand_vel = data.cvel[sand_id][:3]
                    
                    # Calculate relative velocity between foot and sand
                    rel_vel = foot_vel - sand_vel
                    rel_vel_mag = np.linalg.norm(rel_vel)
                    
                    # ====== FRICTION-BASED FORCE ======
                    # Force from foot friction dragging sand backward (opposite to foot motion)
                    if rel_vel_mag > 1e-6:
                        # Normal force magnitude (approximate from distance to foot)
                        dist_to_foot = np.linalg.norm(sand_pos - foot_pos)
                        # Higher normal force = more friction force
                        normal_force = max(1.0, 10.0 - dist_to_foot * 100)  # Decreases with distance
                        
                        # Friction force opposes relative motion
                        friction_force = -sand_friction_coeff * normal_force * (rel_vel / rel_vel_mag)
                    else:
                        friction_force = np.array([0.0, 0.0, 0.0])
                    
                    # ====== FOOT SINKING EFFECT ======
                    # Sand directly under foot gets pushed down
                    push_down_force = 0.0
                    if abs(sand_pos[0] - foot_pos[0]) < 0.01 and abs(sand_pos[1] - foot_pos[1]) < 0.01:
                        # Sand is directly under foot - push downward
                        sand_depth = foot_pos[2] - sand_pos[2]
                        if 0 < sand_depth < foot_sink_depth:
                            # Proportional pushing based on depth
                            push_down_force = (sand_depth / foot_sink_depth) * 3.0
                    
                    # ====== BACKWARD PUSH (Sand displacement behind foot) ======
                    # Sand behind the foot (in negative X from foot) gets pushed backward
                    push_back_force = 0.0
                    relative_x = sand_pos[0] - foot_pos[0]
                    
                    if relative_x < 0.01 and relative_x > -0.04:  # Sand slightly behind foot
                        # Push force magnitude increases closer to foot
                        proximity_factor = 1.0 - (abs(relative_x) / 0.04)
                        push_back_force = -2.0 * proximity_factor  # Push in -X direction
                    
                    # ====== SAND-SAND COUPLING (particles drag each other) ======
                    # Sand in contact with foot experiences drag coefficient
                    drag_force = -sand_drag_coeff * sand_vel
                    
                    # ====== APPLY ALL FORCES TO SAND ======
                    data.xfrc_applied[sand_id, 0] += friction_force[0] + push_back_force + drag_force[0]
                    data.xfrc_applied[sand_id, 1] += friction_force[1] + drag_force[1]
                    data.xfrc_applied[sand_id, 2] += friction_force[2] - push_down_force + drag_force[2]
            
            # ====== SAND-SAND FRICTION (nearby particles interact) ======
            # For particles not in direct foot contact, apply inter-particle forces
            for sand_id in range(1, 1001):
                if sand_id not in contacted_sand:
                    # Check if near other moving sand particles
                    sand_pos = data.xpos[sand_id]
                    sand_vel = data.cvel[sand_id][:3]
                    
                    # Apply damping to reduce unrealistic jittering
                    internal_damping = 0.5  # Light damping on all particles
                    data.xfrc_applied[sand_id, 0] -= internal_damping * sand_vel[0]
                    data.xfrc_applied[sand_id, 1] -= internal_damping * sand_vel[1]
                    data.xfrc_applied[sand_id, 2] -= internal_damping * sand_vel[2]
            
            # Step physics simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            v.sync()
            
            # Progress reporting
            count += 1
            if count % 200 == 0:
                step_in_cycle = t_in_cycle / gait_period
                current_step = int(data.time / gait_period) + 1
                percent_complete = (data.time / total_sim_time) * 100
                print(f"[Step {current_step:2d}/{num_steps}] t={data.time:7.1f}s | Progress: {percent_complete:5.1f}% | Base X={data.qpos[0]:7.4f}m")
    
    # Final summary
    final_x = data.qpos[0]
    steps_completed = int(data.time / gait_period)
    print(f"\n[OK] Walking simulation completed!")
    print(f"[+] Steps completed: {steps_completed}/{num_steps}")
    print(f"[+] Total simulation time: {data.time:.2f}s")
    print(f"[+] Final base X position: {final_x:.4f}m")
    print(f"[+] Distance traveled: {final_x:.4f}m")
    print(f"[+] Average distance per step: {final_x/max(steps_completed, 1):.4f}m")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
