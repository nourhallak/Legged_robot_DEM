#!/usr/bin/env python3
"""
Simple aggressive walking gait for sand - no viewer, just data logging
"""
import mujoco as mj
import numpy as np

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("\n" + "=" * 90)
print(" " * 15 + "AGGRESSIVE SAND WALKING - STRONG BACKWARD PUSH")
print("=" * 90)

mj.mj_resetData(model, data)

# Set initial pose - standing
data.qpos[model.joint("hip_link_2_1").id] = 0.0
data.qpos[model.joint("link_2_1_link_1_1").id] = -0.5
data.qpos[model.joint("link_1_1_foot_1").id] = 0.0

data.qpos[model.joint("hip_link_2_2").id] = 0.0
data.qpos[model.joint("link_2_2_link_1_2").id] = -0.5
data.qpos[model.joint("link_1_2_foot_2").id] = 0.0

mj.mj_forward(model, data)

print(f"[+] Robot standing position X={data.body('hip').xpos[0]:.4f}m, Y={data.body('hip').xpos[1]:.4f}m")

# Simple gait parameters
gait_period = 10.0
sim_time = 0.0
max_time = 20.0  # Reduced from 50 to 20 seconds for faster testing

def get_gait_position(t, phase_offset=0):
    """Generate aggressive stepping gait with strong backward push"""
    t_phase = ((t + phase_offset) % gait_period) / gait_period
    
    if t_phase < 0.5:
        stance_progress = t_phase / 0.5
        # AGGRESSIVE: Push foot back hard (large hip angle)
        hip_angle = 1.2 * np.sin(stance_progress * np.pi)
    else:
        swing_progress = (t_phase - 0.5) / 0.5
        # Swing leg forward for next step
        hip_angle = -1.2 * np.sin(swing_progress * np.pi)
    
    # More aggressive knee motion
    knee_angle = -0.8 + 0.4 * np.sin(t_phase * 2 * np.pi)
    # Ankle for balance
    ankle_angle = 0.3 * np.sin(t_phase * 2 * np.pi)
    
    return hip_angle, knee_angle, ankle_angle

# Strong control parameters
Kp = 1000.0
Kd = 100.0

print("[+] Starting simulation...")
print("[+] Control: Kp=1000, Kd=100 (very aggressive)")
print()

step_count = 0
while sim_time < max_time:
    # Get target positions
    left_hip_target, left_knee_target, left_ankle_target = get_gait_position(sim_time, phase_offset=0)
    right_hip_target, right_knee_target, right_ankle_target = get_gait_position(sim_time, phase_offset=np.pi)
    
    # PD control for left leg
    left_hip_error = left_hip_target - data.qpos[model.joint("hip_link_2_1").id]
    left_hip_vel_error = 0 - data.qvel[model.joint("hip_link_2_1").id]
    data.ctrl[model.actuator("hip_link_2_1_motor").id] = Kp * left_hip_error + Kd * left_hip_vel_error
    
    left_knee_error = left_knee_target - data.qpos[model.joint("link_2_1_link_1_1").id]
    left_knee_vel_error = 0 - data.qvel[model.joint("link_2_1_link_1_1").id]
    data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = Kp * left_knee_error + Kd * left_knee_vel_error
    
    left_ankle_error = left_ankle_target - data.qpos[model.joint("link_1_1_foot_1").id]
    left_ankle_vel_error = 0 - data.qvel[model.joint("link_1_1_foot_1").id]
    data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = Kp * left_ankle_error + Kd * left_ankle_vel_error
    
    # PD control for right leg
    right_hip_error = right_hip_target - data.qpos[model.joint("hip_link_2_2").id]
    right_hip_vel_error = 0 - data.qvel[model.joint("hip_link_2_2").id]
    data.ctrl[model.actuator("hip_link_2_2_motor").id] = Kp * right_hip_error + Kd * right_hip_vel_error
    
    right_knee_error = right_knee_target - data.qpos[model.joint("link_2_2_link_1_2").id]
    right_knee_vel_error = 0 - data.qvel[model.joint("link_2_2_link_1_2").id]
    data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = Kp * right_knee_error + Kd * right_knee_vel_error
    
    right_ankle_error = right_ankle_target - data.qpos[model.joint("link_1_2_foot_2").id]
    right_ankle_vel_error = 0 - data.qvel[model.joint("link_1_2_foot_2").id]
    data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = Kp * right_ankle_error + Kd * right_ankle_vel_error
    
    # Step simulation
    mj.mj_step(model, data)
    sim_time = data.time
    
    # Print status every 5 seconds
    step_count += 1
    if step_count % 500 == 0:
        gait_pct = ((sim_time % gait_period) / gait_period) * 100
        print(f"[t={sim_time:6.2f}s] X={data.body('hip').xpos[0]:.4f}m | Y={data.body('hip').xpos[1]:+.6f}m | Gait={gait_pct:5.1f}%")

print(f"\n[+] Simulation completed!")
print(f"[+] Final position: X={data.body('hip').xpos[0]:.4f}m, Y={data.body('hip').xpos[1]:.6f}m")
print(f"[+] Total displacement: {data.body('hip').xpos[0] - 0.150:.4f}m forward")
