#!/usr/bin/env python3
"""
Biped walking with alternating leg motion (phase offset) for forward movement
Reduced friction allows feet to push sand effectively
"""

import numpy as np
import mujoco
import mujoco.viewer
from scipy.interpolate import interp1d

def main():
    # Load model with sand
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
    
    # Scale amplitudes - much more conservative to prevent instability
    amplitude_scale = 0.05  # Much lower than 25% to prevent violent jerks
    ik_left_hip = ik_left_hip * amplitude_scale
    ik_left_knee = ik_left_knee * amplitude_scale
    ik_left_ankle = ik_left_ankle * amplitude_scale
    ik_right_hip = ik_right_hip * amplitude_scale
    ik_right_knee = ik_right_knee * amplitude_scale
    ik_right_ankle = ik_right_ankle * amplitude_scale
    
    # Interpolations
    interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
    interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
    interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
    interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
    interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
    interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')
    
    # Control gains - very conservative
    Kp = 20.0       # Very low to prevent instability
    Kd = 2.0        # Very low damping
    gait_period = 3.0
    
    # Get body IDs
    try:
        foot_left_id = model.body("foot_1").id
        foot_right_id = model.body("foot_2").id
    except:
        foot_left_id = None
        foot_right_id = None
    
    hip_id = model.body("hip").id
    
    print("=" * 100)
    print("ALTERNATING GAIT TEST - FORWARD WALKING ON SAND")
    print("=" * 100)
    print(f"Strategy: Alternating leg motions (5% amplitude) with 50% phase offset")
    print(f"Sand friction: 0.4 (reduced) | Foot friction: 0.3")
    print(f"Joint control: Kp={Kp}, Kd={Kd} (very conservative)")
    print(f"Forward force: 500-2000N (gradual increase), Rotation damping: 40.0")
    print("=" * 100)
    print()
    
    step_count = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.distance = 1.5
        viewer.cam.elevation = 0
        viewer.cam.lookat[:] = [0.3, 0, 0.5]
        
        while viewer.is_running():
            # Time in gait cycle
            t_cycle = data.time % gait_period
            
            # RIGHT LEG: Normal trajectory
            data.ctrl[3] = Kp * (interp_right_hip(t_cycle) - data.qpos[6]) - Kd * data.qvel[6]
            data.ctrl[4] = Kp * (interp_right_knee(t_cycle) - data.qpos[7]) - Kd * data.qvel[7]
            data.ctrl[5] = Kp * (interp_right_ankle(t_cycle) - data.qpos[8]) - Kd * data.qvel[8]
            
            # LEFT LEG: Phase shifted by 50% (alternating motion)
            # When right leg pushes, left leg swings, and vice versa
            phase_offset = gait_period * 0.5
            t_left = (data.time + phase_offset) % gait_period
            
            data.ctrl[0] = Kp * (interp_left_hip(t_left) - data.qpos[3]) - Kd * data.qvel[3]
            data.ctrl[1] = Kp * (interp_left_knee(t_left) - data.qpos[4]) - Kd * data.qvel[4]
            data.ctrl[2] = Kp * (interp_left_ankle(t_left) - data.qpos[5]) - Kd * data.qvel[5]
            
            # Forward force - gradually increase
            if data.time < 5.0:
                forward_force = 500.0  # Start gentle
            elif data.time < 10.0:
                forward_force = 1000.0
            else:
                forward_force = 2000.0
            data.xfrc_applied[hip_id, 0] = forward_force
            
            # Rotation control - prevent spinning
            rotation_kp = 40.0
            rotation_kd = 20.0
            torque_z = rotation_kp * (0.0 - data.qpos[2]) - rotation_kd * data.qvel[2]
            data.xfrc_applied[hip_id, 5] = torque_z
            
            # Lateral control - keep centered on sand
            lateral_kp = 100.0
            lateral_kd = 20.0
            lateral_force = lateral_kp * (0.0 - data.qpos[1]) - lateral_kd * data.qvel[1]
            data.xfrc_applied[hip_id, 1] = lateral_force
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            step_count += 1
            
            if step_count % 100 == 0:
                hip_x = data.xpos[hip_id][0]
                hip_y = data.xpos[hip_id][1]
                
                left_z = data.xpos[foot_left_id][2] if foot_left_id else 0
                right_z = data.xpos[foot_right_id][2] if foot_right_id else 0
                
                distance = hip_x - 0.150
                status_left = "ON" if 0.442 <= left_z <= 0.460 else "OFF"
                status_right = "ON" if 0.442 <= right_z <= 0.460 else "OFF"
                
                print(f"T:{data.time:6.2f}s | Hip X:{hip_x:+7.4f}m | Dist:{distance:+7.4f}m | Feet Z: L={left_z:.4f}m ({status_left:3s}) | R={right_z:.4f}m ({status_right:3s})")

if __name__ == "__main__":
    main()

