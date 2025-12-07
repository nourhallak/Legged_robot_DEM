#!/usr/bin/env python3
"""
Robot walking simulation using IK-computed joint angles.
UPDATED for realistic trajectories with detailed debugging.
"""

import numpy as np
import mujoco
from mujoco import viewer
from scipy.interpolate import interp1d

def main():
    """Simulate robot walking with IK-computed trajectories."""
    
    print("\n" + "="*80)
    print("BIPEDAL ROBOT WALKING WITH IK-COMPUTED REALISTIC TRAJECTORIES")
    print("="*80 + "\n")
    
    # Load model
    model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
    data = mujoco.MjData(model)
    print("[+] Loaded robot model: legged_robot_ik.xml")
    
    # Load trajectory data to verify
    traj_times = np.load("traj_times.npy")
    traj_left_foot = np.load("traj_left_foot.npy")
    traj_right_foot = np.load("traj_right_foot.npy")
    print(f"[+] Loaded planned trajectories: {len(traj_times)} points")
    print(f"    - Left foot X range: {traj_left_foot[:, 0].min():.4f} to {traj_left_foot[:, 0].max():.4f} m")
    print(f"    - Right foot X range: {traj_right_foot[:, 0].min():.4f} to {traj_right_foot[:, 0].max():.4f} m")
    
    # Load IK solutions
    try:
        ik_times = np.load("ik_times.npy")
        ik_left_hip = np.load("ik_left_hip.npy")
        ik_left_knee = np.load("ik_left_knee.npy")
        ik_left_ankle = np.load("ik_left_ankle.npy")
        ik_right_hip = np.load("ik_right_hip.npy")
        ik_right_knee = np.load("ik_right_knee.npy")
        ik_right_ankle = np.load("ik_right_ankle.npy")
        traj_base_pos = np.load("traj_base_pos.npy")
        
        print(f"[+] Loaded IK solutions: {len(ik_times)} points")
        print(f"    - Left hip angle range: {ik_left_hip.min():.4f} to {ik_left_hip.max():.4f} rad")
        print(f"    - Left knee angle range: {ik_left_knee.min():.4f} to {ik_left_knee.max():.4f} rad")
        print(f"    - Left ankle angle range: {ik_left_ankle.min():.4f} to {ik_left_ankle.max():.4f} rad")
    except FileNotFoundError as e:
        print(f"[-] ERROR: Missing file: {e}")
        print("[-] Run: python trajectory_planning.py")
        print("[-] Then: python solve_ik.py")
        return
    
    # Create interpolation functions for smooth playback
    print("\n[+] Creating cubic spline interpolations...")
    
    interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
    interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
    interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
    
    interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
    interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
    interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')
    
    interp_base_x = interp1d(ik_times, traj_base_pos[:, 0], kind='cubic', fill_value='extrapolate')
    interp_base_y = interp1d(ik_times, traj_base_pos[:, 1], kind='cubic', fill_value='extrapolate')
    interp_base_z = interp1d(ik_times, traj_base_pos[:, 2], kind='cubic', fill_value='extrapolate')
    
    print("[+] Interpolation functions created")
    
    # Control parameters
    Kp = 300.0  # Proportional gain
    Kd = 30.0   # Derivative gain
    
    # Base motion control
    base_Kp = 5000.0  # Base position gain (increased for more forward motion)
    base_Kd = 50.0   # Base velocity damping (to prevent oscillation)
    
    # Get cycle parameters
    # Gait period is 50 seconds (one complete walk cycle)
    cycle_duration = 50.0
    max_time = ik_times[-1]  # Total time available
    
    print(f"\n[+] Walking parameters:")
    print(f"    - Gait cycle duration: {cycle_duration:.2f}s")
    print(f"    - Total time available: {max_time:.2f}s")
    print(f"    - Control gains: Kp={Kp}, Kd={Kd}")
    print(f"    - Simulation timestep: {model.opt.timestep}s")
    
    print(f"\n[+] Starting MuJoCo viewer...")
    print(f"[+] Close the window to stop walking\n")
    
    # Initialize viewer with passive mode (lets physics run while viewer displays)
    with viewer.launch_passive(model, data) as v:
        v.cam.azimuth = 90  # Side view
        v.cam.distance = 1.5  # Camera distance from robot
        v.cam.elevation = 0  # View from side
        v.cam.lookat[:] = [0, 0, 0.5]  # Look at robot center
        
        # Main simulation loop
        count = 0
        total_cycles = 0
        
        while v.is_running():
            # Calculate time within current cycle
            t_in_cycle = data.time % cycle_duration
            
            # Get desired joint angles from IK
            des_left_hip = interp_left_hip(t_in_cycle)
            des_left_knee = interp_left_knee(t_in_cycle)
            des_left_ankle = interp_left_ankle(t_in_cycle)
            
            des_right_hip = interp_right_hip(t_in_cycle)
            des_right_knee = interp_right_knee(t_in_cycle)
            des_right_ankle = interp_right_ankle(t_in_cycle)
            
            # Add forward velocity to base: 20 steps × ~0.03m per step = 0.6m total distance
            # Over 100 seconds of simulation = 0.006 m/s forward velocity
            target_base_x = data.time * 0.006  # Forward velocity: 6mm/s
            
            # Set joint actuators with PD control
            # Left leg (joints 3, 4, 5)
            data.ctrl[0] = Kp * (des_left_hip - data.qpos[3]) - Kd * data.qvel[3]
            data.ctrl[1] = Kp * (des_left_knee - data.qpos[4]) - Kd * data.qvel[4]
            data.ctrl[2] = Kp * (des_left_ankle - data.qpos[5]) - Kd * data.qvel[5]
            
            # Right leg (joints 6, 7, 8)
            data.ctrl[3] = Kp * (des_right_hip - data.qpos[6]) - Kd * data.qvel[6]
            data.ctrl[4] = Kp * (des_right_knee - data.qpos[7]) - Kd * data.qvel[7]
            data.ctrl[5] = Kp * (des_right_ankle - data.qpos[8]) - Kd * data.qvel[8]
            
            # Apply external force to push base forward and damp oscillations
            # This helps overcome the backward drag from leg swings
            forward_force = base_Kp * (target_base_x - data.qpos[0]) - base_Kd * data.qvel[0]
            data.xfrc_applied[0, 0] = forward_force  # Apply force to hip body (body 0)
            
            # Damp rotation in place: zero out angular velocity around Z axis
            # This prevents unwanted spinning from asymmetric leg movements
            # Apply counter-torque to stabilize rotation
            rotation_damping = 50.0  # Damping coefficient for rotational motion
            data.xfrc_applied[0, 5] = -rotation_damping * data.qvel[2]  # Torque around Z axis
            
            # Step physics simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            v.sync()
            
            # Progress reporting
            count += 1
            if count % 200 == 0:
                step_in_cycle = t_in_cycle / cycle_duration
                total_cycles = int(data.time / cycle_duration)
                print(f"[t] t={data.time:7.2f}s | Cycle #{total_cycles} | Progress in cycle: {step_in_cycle*100:5.1f}% | Base X={data.qpos[0]:7.4f}m")
    
    # Final summary
    final_x = data.qpos[0]
    total_cycles_completed = int(data.time / cycle_duration)
    print(f"\n[OK] Walking simulation stopped!")
    print(f"[+] Total cycles completed: {total_cycles_completed}")
    print(f"[+] Total simulation time: {data.time:.2f}s")
    print(f"[+] Final base X position: {final_x:.4f}m")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
