#!/usr/bin/env python3
"""
Aggressive knee-driven gait: Use knee extension to push sand backward
This creates reaction forces that propel robot forward
"""
import mujoco as mj
import numpy as np

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("\n" + "=" * 90)
print(" " * 15 + "KNEE-DRIVEN WALKING - AGGRESSIVE KNEE EXTENSION")
print("=" * 90)

mj.mj_resetData(model, data)

# Set initial pose
data.qpos[model.joint("hip_link_2_1").id] = 0.0
data.qpos[model.joint("link_2_1_link_1_1").id] = -0.5
data.qpos[model.joint("link_1_1_foot_1").id] = 0.0

data.qpos[model.joint("hip_link_2_2").id] = 0.0
data.qpos[model.joint("link_2_2_link_1_2").id] = -0.5
data.qpos[model.joint("link_1_2_foot_2").id] = 0.0

mj.mj_forward(model, data)

print(f"[+] Robot standing position X={data.body('hip').xpos[0]:.4f}m, Y={data.body('hip').xpos[1]:.4f}m")

gait_period = 3.0
sim_time = 0.0
max_time = 60.0

def get_knee_driven_gait(t, phase_offset=0):
    """
    Knee-driven gait: Rapid knee extension pushes sand backward
    Hip and ankle provide stability
    """
    t_phase = ((t + phase_offset) % gait_period) / gait_period
    
    # PUSH PHASE (0-0.4): Rapidly extend knee
    if t_phase < 0.4:
        progress = t_phase / 0.4
        # Keep hip steady in middle position
        hip_angle = 0.3 * np.sin(progress * np.pi * 2)  # Small oscillation for stability
        # AGGRESSIVE knee extension - rapid snap
        knee_angle = -1.8 + 1.5 * progress  # From -1.8 (very bent) to -0.3 (extended)
        # Ankle locked
        ankle_angle = 0.0
        
    # SWING PHASE (0.4-1.0): Bring leg back, prepare for next push
    else:
        progress = (t_phase - 0.4) / 0.6
        # Hip stabilizes
        hip_angle = 0.3 * np.sin(progress * np.pi * 2)
        # Flex knee back for next push
        knee_angle = -0.3 - 1.5 * progress  # Return to bent position
        ankle_angle = 0.0
    
    return hip_angle, knee_angle, ankle_angle

# AGGRESSIVE control
Kp = 1500.0
Kd = 150.0

print("[+] Starting simulation...")
print(f"[+] Gait: Knee-driven (rapid extension to push sand)")
print(f"[+] Control: Kp={Kp}, Kd={Kd} (very stiff)")
print()

while sim_time < max_time:
    # Left leg (phase 0)
    left_hip_target, left_knee_target, left_ankle_target = get_knee_driven_gait(sim_time, phase_offset=0)
    
    # Right leg (180° out of phase)
    right_hip_target, right_knee_target, right_ankle_target = get_knee_driven_gait(sim_time, phase_offset=gait_period/2)
    
    # Apply PD control
    left_hip_error = left_hip_target - data.qpos[model.joint("hip_link_2_1").id]
    left_hip_vel_error = 0 - data.qvel[model.joint("hip_link_2_1").id]
    data.ctrl[model.actuator("hip_link_2_1_motor").id] = Kp * left_hip_error + Kd * left_hip_vel_error
    
    left_knee_error = left_knee_target - data.qpos[model.joint("link_2_1_link_1_1").id]
    left_knee_vel_error = 0 - data.qvel[model.joint("link_2_1_link_1_1").id]
    data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = Kp * left_knee_error + Kd * left_knee_vel_error
    
    left_ankle_error = left_ankle_target - data.qpos[model.joint("link_1_1_foot_1").id]
    left_ankle_vel_error = 0 - data.qvel[model.joint("link_1_1_foot_1").id]
    data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = Kp * left_ankle_error + Kd * left_ankle_vel_error
    
    right_hip_error = right_hip_target - data.qpos[model.joint("hip_link_2_2").id]
    right_hip_vel_error = 0 - data.qvel[model.joint("hip_link_2_2").id]
    data.ctrl[model.actuator("hip_link_2_2_motor").id] = Kp * right_hip_error + Kd * right_hip_vel_error
    
    right_knee_error = right_knee_target - data.qpos[model.joint("link_2_2_link_1_2").id]
    right_knee_vel_error = 0 - data.qvel[model.joint("link_2_2_link_1_2").id]
    data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = Kp * right_knee_error + Kd * right_knee_vel_error
    
    right_ankle_error = right_ankle_target - data.qpos[model.joint("link_1_2_foot_2").id]
    right_ankle_vel_error = 0 - data.qvel[model.joint("link_1_2_foot_2").id]
    data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = Kp * right_ankle_error + Kd * right_ankle_vel_error
    
    # Step
    mj.mj_step(model, data)
    sim_time += model.opt.timestep
    
    # Print
    if int(sim_time) > int(sim_time - model.opt.timestep):
        gait_phase = (sim_time % gait_period) / gait_period * 100
        x_pos = data.body('hip').xpos[0]
        y_pos = data.body('hip').xpos[1]
        x_vel = data.body('hip').cvel[0]
        print(f"[t={sim_time:6.2f}s] X={x_pos:.4f}m | Y={y_pos:+.6f}m | VelX={x_vel:+.5f}m/s | Phase={gait_phase:5.1f}%")

print()
print("[+] Simulation completed!")
start_x = 0.150
final_x = data.body('hip').xpos[0]
displacement = final_x - start_x
print(f"[+] Final position: X={final_x:.4f}m, Y={data.body('hip').xpos[1]:+.6f}m")
print(f"[+] Total displacement: {displacement:.4f}m")
if displacement > 0:
    print(f"[+] ✓ ROBOT IS WALKING FORWARD!")
else:
    print(f"[+] ✗ Robot walked backward")
print()
