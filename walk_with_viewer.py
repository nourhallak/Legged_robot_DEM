#!/usr/bin/env python3
"""
Interactive 3D visualization of robot walking on sand using MuJoCo viewer
Shows the complete walking motion with sand deformation in real-time
"""

import numpy as np
import mujoco
from scipy.interpolate import interp1d
import mujoco.viewer

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
    
    # Get IDs
    foot1_id = model.body("foot_1").id
    foot2_id = model.body("foot_2").id
    
    # Initial sand positions
    sand_pos_init = data.xpos[1:1001].copy()
    
    total_contacts = 0
    
    def physics_callback(model, data):
        """Physics callback - called at each simulation step"""
        nonlocal total_contacts
        
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
    
    # Create viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n" + "="*70)
        print("INTERACTIVE 3D VISUALIZATION: ROBOT WALKING ON SAND")
        print("="*70)
        print("\nControls:")
        print("  Mouse: Rotate view")
        print("  Scroll: Zoom in/out")
        print("  Right-click drag: Pan")
        print("  'R': Reset view")
        print("  'F': Toggle frame-by-frame (Ctrl+E after pausing)")
        print("  Close window to end simulation")
        print("\nWatch the robot walk forward while sand particles deform beneath feet")
        print("="*70 + "\n")
        
        # Run simulation
        while viewer.is_running() and data.time < 30.0:
            physics_callback(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()
    
    # Final analysis
    sand_pos_final = data.xpos[1:1001]
    sand_disp = np.linalg.norm(sand_pos_final - sand_pos_init, axis=1)
    sand_displaced = np.sum(sand_disp > 0.001)
    hip_id = model.body("hip").id
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Total simulation time:      {data.time:.2f}s")
    print(f"Robot base X position:      {data.xpos[hip_id][0]:+.4f}m")
    print(f"Total foot-sand contacts:   {total_contacts:,}")
    print(f"Sand particles displaced:   {sand_displaced} / 1000")
    print(f"Average displacement:       {np.mean(sand_disp)*1000:.2f}mm")
    print(f"Maximum displacement:       {np.max(sand_disp)*1000:.2f}mm")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
