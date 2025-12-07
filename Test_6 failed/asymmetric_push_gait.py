import mujoco as mj
import numpy as np

# Load model
model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

# Find joint indices
def find_joint_id(name):
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)

# Joint IDs
root_x_id = find_joint_id("root_x")
root_y_id = find_joint_id("root_y")
left_hip_id = find_joint_id("left_hip")
left_knee_id = find_joint_id("left_knee")
left_ankle_id = find_joint_id("left_ankle")
right_hip_id = find_joint_id("right_hip")
right_knee_id = find_joint_id("right_knee")
right_ankle_id = find_joint_id("right_ankle")

# Actuator indices
left_hip_act = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "left_hip_actuator")
left_knee_act = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "left_knee_actuator")
left_ankle_act = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "left_ankle_actuator")
right_hip_act = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "right_hip_actuator")
right_knee_act = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "right_knee_actuator")
right_ankle_act = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "right_ankle_actuator")

print("="*80)
print("ASYMMETRIC PUSH GAIT - STRONG FORWARD BIAS")
print("="*80)
print(f"[+] Gait: asymmetric with strong backward push on ground")
print(f"[+] Y constraint: ±0.02m, damping=20.0")
print(f"[+] Sand physics: density=0.1, friction=0.00001, timestep=0.002s")

# PD control parameters
Kp = 1000
Kd = 100
gait_period = 10.0
max_time = 20.0

def get_asymmetric_gait(t, phase_offset=0):
    """Generate aggressive stepping gait with strong backward push"""
    t_phase = ((t + phase_offset) % gait_period) / gait_period
    
    if t_phase < 0.5:
        stance_progress = t_phase / 0.5
        hip_angle = 1.2 * np.sin(stance_progress * np.pi)  # Strong push
    else:
        swing_progress = (t_phase - 0.5) / 0.5
        hip_angle = -1.2 * np.sin(swing_progress * np.pi)  # Swing forward
    
    knee_angle = -0.8 + 0.4 * np.cos(t_phase * 2 * np.pi)
    ankle_angle = 0.3 * np.sin(t_phase * 2 * np.pi)
    
    return hip_angle, knee_angle, ankle_angle

# Simulation loop
print(f"[+] Starting 20-second simulation...")
step_count = 0
while data.time < max_time:
    # Get desired positions
    left_hip_des, left_knee_des, left_ankle_des = get_asymmetric_gait(data.time, 0)
    right_hip_des, right_knee_des, right_ankle_des = get_asymmetric_gait(data.time, np.pi)
    
    # PD control
    left_hip_err = left_hip_des - data.qpos[left_hip_id]
    left_knee_err = left_knee_des - data.qpos[left_knee_id]
    left_ankle_err = left_ankle_des - data.qpos[left_ankle_id]
    
    right_hip_err = right_hip_des - data.qpos[right_hip_id]
    right_knee_err = right_knee_des - data.qpos[right_knee_id]
    right_ankle_err = right_ankle_des - data.qpos[right_ankle_id]
    
    data.ctrl[left_hip_act] = Kp * left_hip_err - Kd * data.qvel[left_hip_id]
    data.ctrl[left_knee_act] = Kp * left_knee_err - Kd * data.qvel[left_knee_id]
    data.ctrl[left_ankle_act] = Kp * left_ankle_err - Kd * data.qvel[left_ankle_id]
    
    data.ctrl[right_hip_act] = Kp * right_hip_err - Kd * data.qvel[right_hip_id]
    data.ctrl[right_knee_act] = Kp * right_knee_err - Kd * data.qvel[right_knee_id]
    data.ctrl[right_ankle_act] = Kp * right_ankle_err - Kd * data.qvel[right_ankle_id]
    
    # Step
    mj.mj_step(model, data)
    step_count += 1
    
    # Print every 1 second
    if int(data.time) % 1 == 0 and step_count % 500 == 0:
        gait_progress = ((data.time % gait_period) / gait_period) * 100
        print(f"[t={data.time:5.2f}s] X={data.xpos[1,0]:6.4f}m | Y={data.xpos[1,1]:+8.6f}m | Gait={gait_progress:5.1f}%")

print(f"\n[+] Simulation completed!")
print(f"[+] Final position: X={data.xpos[1,0]:.4f}m, Y={data.xpos[1,1]:+.6f}m")
print(f"[+] Total displacement: {data.xpos[1,0]-0.150:.4f}m forward")
print(f"[+] Speed: {(data.xpos[1,0]-0.150)/20:.4f} m/s")
