#!/usr/bin/env python3
"""
Asymmetric forward walking - left leg drives, right leg follows
"""
import mujoco as mj
import numpy as np

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("\n" + "=" * 90)
print(" " * 15 + "ASYMMETRIC DRIVING GAIT - LEFT LEG DRIVES FORWARD")
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

gait_period = 5.0
sim_time = 0.0
max_time = 50.0  

def get_left_leg_driving(t):
    """
    LEFT LEG: Drives forward with strong push and recovery
    """
    t_phase = (t % gait_period) / gait_period
    
    if t_phase < 0.5:
        # STANCE: Push backward hard (negative hip = backward extension)
        progress = t_phase / 0.5
        hip_angle = -0.8 * np.sin(progress * np.pi)
        knee_angle = -0.3 - 0.25 * np.sin(progress * np.pi)
        ankle_angle = 0.2
    else:
        # SWING: Bring leg forward for next step
        progress = (t_phase - 0.5) / 0.5
        hip_angle = -0.8 + 1.6 * progress
        knee_angle = -0.55 + 0.4 * np.sin(progress * np.pi)
        ankle_angle = 0.1
    
    return hip_angle, knee_angle, ankle_angle

def get_right_leg_support(t):
    """
    RIGHT LEG: Provides support, lighter push
    Offset by 0.25 of period (quarter phase)
    """
    t_phase = ((t + gait_period * 0.25) % gait_period) / gait_period
    
    if t_phase < 0.5:
        # LIGHTER STANCE: Push but not as hard as left
        progress = t_phase / 0.5
        hip_angle = -0.5 * np.sin(progress * np.pi)  # Increased from -0.3
        knee_angle = -0.3 - 0.2 * np.sin(progress * np.pi)  # Increased push
        ankle_angle = 0.2
    else:
        # SWING: Follow the left leg
        progress = (t_phase - 0.5) / 0.5
        hip_angle = -0.5 + 1.0 * progress  # Increased from -0.3 + 0.6
        knee_angle = -0.5 + 0.35 * np.sin(progress * np.pi)
        ankle_angle = 0.1
    
    return hip_angle, knee_angle, ankle_angle

# Control parameters
Kp = 1000.0
Kd = 100.0

print("[+] Starting simulation...")
print(f"[+] Gait period: {gait_period}s, Asymmetric driving (left leg dominates)")
print(f"[+] Control: Kp={Kp}, Kd={Kd}")
print()

while sim_time < max_time:
    # Left leg drives
    left_hip_target, left_knee_target, left_ankle_target = get_left_leg_driving(sim_time)
    
    # Right leg supports
    right_hip_target, right_knee_target, right_ankle_target = get_right_leg_support(sim_time)
    
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
