#!/usr/bin/env python3
"""
FIXED GAIT: Knees FLEX HARD to push feet into sand
Then extend to propel robot forward
"""
import mujoco as mj
import numpy as np

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("\n" + "=" * 90)
print(" " * 15 + "CORRECTED GAIT - KNEES FLEX INTO SAND, THEN EXTEND")
print("=" * 90)

mj.mj_resetData(model, data)

# Initial standing pose - knees EXTREMELY BENT to touch sand
data.qpos[model.joint("hip_link_2_1").id] = 0.0
data.qpos[model.joint("link_2_1_link_1_1").id] = -2.4  # Maximum bend to contact sand
data.qpos[model.joint("link_1_1_foot_1").id] = 0.0

data.qpos[model.joint("hip_link_2_2").id] = 0.0
data.qpos[model.joint("link_2_2_link_1_2").id] = -2.4  # Maximum bend to contact sand
data.qpos[model.joint("link_1_2_foot_2").id] = 0.0

mj.mj_forward(model, data)

hip_pos = data.body('hip').xpos
foot1_pos = data.body('foot_1').xpos

print(f"[+] Initial standing with bent knees")
print(f"    Hip position: Z={hip_pos[2]:.4f}m")
print(f"    Foot position: Z={foot1_pos[2]:.4f}m (sand is at Z=0.442m)")

gait_period = 4.0
sim_time = 0.0
max_time = 60.0

def get_contact_gait(t, phase_offset=0):
    """
    Gait: 
    1. FLEX: Bend knee deeply to contact and load sand (-2.4 rad)
    2. EXTEND: Rapidly extend knee to push robot forward (-0.8 rad)
    3. SWING: Bring leg back for next step
    """
    t_phase = ((t + phase_offset) % gait_period) / gait_period
    
    # FLEX & LOAD (0-0.3): Contact sand, bend deeper
    if t_phase < 0.3:
        progress = t_phase / 0.3
        # Small hip motion for balance
        hip_angle = 0.1 * np.sin(progress * np.pi)
        # FLEX knee into sand - go to -2.4
        knee_angle = -2.3 - 0.2 * np.sin(progress * np.pi)
        ankle_angle = 0.0
        
    # PUSH (0.3-0.6): Rapidly extend to push off
    elif t_phase < 0.6:
        progress = (t_phase - 0.3) / 0.3
        hip_angle = 0.1 * np.sin(progress * np.pi)
        # SNAP extend knee - rapid push from -2.3 to -0.8
        knee_angle = -2.3 + 1.5 * np.sin(progress * np.pi)
        ankle_angle = 0.0
        
    # SWING (0.6-1.0): Relax and prepare for next
    else:
        progress = (t_phase - 0.6) / 0.4
        hip_angle = 0.1 * np.sin(progress * np.pi)
        # Return to bent position
        knee_angle = -0.8 - 1.5 * progress
        ankle_angle = 0.0
    
    return hip_angle, knee_angle, ankle_angle

# Strong control to overcome sand resistance
Kp = 1800.0
Kd = 180.0

print(f"[+] Gait: Contact -> Push -> Swing (alternating legs)")
print(f"[+] Control: Kp={Kp}, Kd={Kd} (very stiff for sand contact)")
print()

contact_count = 0
while sim_time < max_time:
    left_hip_target, left_knee_target, left_ankle_target = get_contact_gait(sim_time, phase_offset=0)
    right_hip_target, right_knee_target, right_ankle_target = get_contact_gait(sim_time, phase_offset=gait_period/2)
    
    # Apply controls
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
    
    # Check for foot-sand contacts
    ncon_before = data.ncon
    
    # Step
    mj.mj_step(model, data)
    sim_time += model.opt.timestep
    
    # Count contacts (simple check)
    if data.ncon > ncon_before:
        contact_count += 1
    
    # Print status
    if int(sim_time) > int(sim_time - model.opt.timestep):
        gait_phase = (sim_time % gait_period) / gait_period * 100
        x_pos = data.body('hip').xpos[0]
        y_pos = data.body('hip').xpos[1]
        x_vel = data.body('hip').cvel[0]
        foot_z = data.body('foot_1').xpos[2]
        
        print(f"[t={sim_time:6.2f}s] X={x_pos:.4f}m | Y={y_pos:+.6f}m | VelX={x_vel:+.5f}m/s | "
              f"Foot_Z={foot_z:.4f}m | Phase={gait_phase:5.1f}%")

print()
print("[+] Simulation completed!")
start_x = 0.150
final_x = data.body('hip').xpos[0]
displacement = final_x - start_x
print(f"[+] Final position: X={final_x:.4f}m, Y={data.body('hip').xpos[1]:+.6f}m")
print(f"[+] Total displacement: {displacement:.4f}m")
print(f"[+] Contact events: {contact_count}")
if displacement > 0.05:
    print(f"[+] SUCCESS: Robot walked forward!")
else:
    print(f"[!] FAILED: Insufficient forward motion")
print()
