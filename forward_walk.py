#!/usr/bin/env python3
"""
Forward walking gait for sand - asymmetric push
"""
import mujoco as mj
import numpy as np

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("\n" + "=" * 90)
print(" " * 20 + "FORWARD WALKING ON SAND - ASYMMETRIC PUSH")
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

# Shorter gait period for snappier steps
gait_period = 4.0
sim_time = 0.0
max_time = 40.0  

def get_forward_gait(t, phase_offset=0):
    """
    Generate asymmetric walking gait that propels forward
    Left and right legs are 180° out of phase
    """
    t_phase = ((t + phase_offset) % gait_period) / gait_period
    
    # STANCE PHASE (0-0.5): Push foot backward to propel forward
    if t_phase < 0.5:
        stance_progress = t_phase / 0.5
        # Push backward by having negative hip angle
        hip_angle = -0.9 * np.sin(stance_progress * np.pi)
        
        # Keep knee relatively extended during stance
        knee_angle = -0.35 - 0.25 * np.sin(stance_progress * np.pi)
        
        # Ankle support
        ankle_angle = 0.25 * np.cos(stance_progress * np.pi)
        
    # SWING PHASE (0.5-1.0): Bring foot forward for next step
    else:
        swing_progress = (t_phase - 0.5) / 0.5
        # Swing leg forward
        hip_angle = -0.9 + 1.8 * swing_progress
        
        # Increase knee angle during swing
        knee_angle = -0.6 + 0.45 * np.sin(swing_progress * np.pi)
        
        # Ankle clearance
        ankle_angle = 0.12
    
    return hip_angle, knee_angle, ankle_angle

# Strong control parameters
Kp = 1200.0
Kd = 120.0

print("[+] Starting simulation...")
print(f"[+] Gait period: {gait_period}s, Asymmetric push-forward design")
print(f"[+] Control: Kp={Kp}, Kd={Kd}")
print()

while sim_time < max_time:
    # Get target positions for left leg (phase 0)
    left_hip_target, left_knee_target, left_ankle_target = get_forward_gait(sim_time, phase_offset=0)
    
    # Get target positions for right leg (180° out of phase)
    right_hip_target, right_knee_target, right_ankle_target = get_forward_gait(sim_time, phase_offset=gait_period/2)
    
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
    
    # Step and advance time
    mj.mj_step(model, data)
    sim_time += model.opt.timestep
    
    # Print status every 1 second
    if int(sim_time) > int(sim_time - model.opt.timestep):
        gait_phase = (sim_time % gait_period) / gait_period * 100
        x_pos = data.body('hip').xpos[0]
        y_pos = data.body('hip').xpos[1]
        print(f"[t={sim_time:6.2f}s] X={x_pos:.4f}m | Y={y_pos:+.6f}m | Phase={gait_phase:5.1f}%")

print()
print("[+] Simulation completed!")
start_x = 0.150
final_x = data.body('hip').xpos[0]
displacement = final_x - start_x
print(f"[+] Final position: X={final_x:.4f}m, Y={data.body('hip').xpos[1]:+.6f}m")
print(f"[+] Total displacement: {displacement:.4f}m forward")
print()
