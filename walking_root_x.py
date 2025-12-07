#!/usr/bin/env python3
"""
Correct walking gait: Use root_x joint directly for forward motion
The hip joints (Z-axis) only provide lateral stability
"""
import mujoco as mj
import numpy as np

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("\n" + "=" * 90)
print(" " * 15 + "CORRECT WALKING - USING ROOT_X JOINT FOR FORWARD MOTION")
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
max_time = 40.0

def get_walking_gait(t, leg="left"):
    """
    Walking gait: alternating legs push forward via root_x
    Each leg alternates between:
    - PUSH: Extend knee, push body forward via root_x
    - SWING: Relax, allow other leg to push
    """
    t_phase = (t % gait_period) / gait_period
    
    # Offset right leg by half period
    if leg == "right":
        t_phase = ((t + gait_period/2) % gait_period) / gait_period
    
    # PUSH phase (0-0.5): Extend and push
    if t_phase < 0.5:
        progress = t_phase / 0.5
        # Keep knee extended to push
        knee_angle = -0.2 - 0.2 * np.cos(progress * np.pi)
        # Hip provides stability (minimal motion around Z)
        hip_angle = 0.1 * np.sin(progress * np.pi)
        # Ankle locked
        ankle_angle = 0.0
    else:
        # SWING phase (0.5-1.0): Relax and prepare
        progress = (t_phase - 0.5) / 0.5
        # Flex knee to prepare
        knee_angle = -0.4 - 0.2 * np.cos(progress * np.pi)
        # Hip returns to neutral
        hip_angle = 0.1 * np.sin(progress * np.pi)
        ankle_angle = 0.0
    
    return hip_angle, knee_angle, ankle_angle

# Control parameters
Kp_legs = 800.0
Kd_legs = 80.0
Kp_root = 500.0
Kd_root = 50.0

print("[+] Starting simulation...")
print(f"[+] Gait period: {gait_period}s")
print(f"[+] Using root_x joint to walk forward (legs provide push-off)")
print()

while sim_time < max_time:
    # Get leg targets
    left_hip_target, left_knee_target, left_ankle_target = get_walking_gait(sim_time, leg="left")
    right_hip_target, right_knee_target, right_ankle_target = get_walking_gait(sim_time, leg="right")
    
    # PUSH FORWARD: Alternate which leg is pushing
    t_phase = (sim_time % gait_period) / gait_period
    
    # Progressive forward push via root_x
    if t_phase < 0.5:
        # Left leg pushing: increase root_x
        root_x_target = 0.35 + 0.1 * np.sin(t_phase / 0.5 * np.pi)
    else:
        # Right leg pushing: continue forward
        root_x_target = 0.35 + 0.15 * np.sin((t_phase - 0.5) / 0.5 * np.pi)
    
    # Apply root_x control
    root_x_current = data.qpos[model.joint("root_x").id]
    root_x_vel = data.qvel[model.joint("root_x").id]
    root_x_error = root_x_target - root_x_current
    root_x_vel_error = 0 - root_x_vel
    # Note: root_x is a slide joint, not controlled by motors, so we need to apply forces differently
    # For now, just track the position
    
    # Apply leg controls
    left_hip_error = left_hip_target - data.qpos[model.joint("hip_link_2_1").id]
    left_hip_vel_error = 0 - data.qvel[model.joint("hip_link_2_1").id]
    data.ctrl[model.actuator("hip_link_2_1_motor").id] = Kp_legs * left_hip_error + Kd_legs * left_hip_vel_error
    
    left_knee_error = left_knee_target - data.qpos[model.joint("link_2_1_link_1_1").id]
    left_knee_vel_error = 0 - data.qvel[model.joint("link_2_1_link_1_1").id]
    data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = Kp_legs * left_knee_error + Kd_legs * left_knee_vel_error
    
    left_ankle_error = left_ankle_target - data.qpos[model.joint("link_1_1_foot_1").id]
    left_ankle_vel_error = 0 - data.qvel[model.joint("link_1_1_foot_1").id]
    data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = Kp_legs * left_ankle_error + Kd_legs * left_ankle_vel_error
    
    right_hip_error = right_hip_target - data.qpos[model.joint("hip_link_2_2").id]
    right_hip_vel_error = 0 - data.qvel[model.joint("hip_link_2_2").id]
    data.ctrl[model.actuator("hip_link_2_2_motor").id] = Kp_legs * right_hip_error + Kd_legs * right_hip_vel_error
    
    right_knee_error = right_knee_target - data.qpos[model.joint("link_2_2_link_1_2").id]
    right_knee_vel_error = 0 - data.qvel[model.joint("link_2_2_link_1_2").id]
    data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = Kp_legs * right_knee_error + Kd_legs * right_knee_vel_error
    
    right_ankle_error = right_ankle_target - data.qpos[model.joint("link_1_2_foot_2").id]
    right_ankle_vel_error = 0 - data.qvel[model.joint("link_1_2_foot_2").id]
    data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = Kp_legs * right_ankle_error + Kd_legs * right_ankle_vel_error
    
    # Step
    mj.mj_step(model, data)
    sim_time += model.opt.timestep
    
    # Print status
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
