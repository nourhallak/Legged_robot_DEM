"""
Walking on sand using root translation + leg control
Uses the root_x joint to move the robot forward while legs provide contact and stability
"""
import mujoco as mj
import numpy as np
from scipy.interpolate import interp1d

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("=" * 80)
print("WALKING ON SAND - Root translation with leg support")
print("=" * 80)

# Load IK
ik_times = np.load('ik_times.npy')
ik_left_hip = np.load('ik_left_hip.npy') * 0.20
ik_left_knee = np.load('ik_left_knee.npy') * 0.20
ik_left_ankle = np.load('ik_left_ankle.npy') * 0.20
ik_right_hip = np.load('ik_right_hip.npy') * 0.20
ik_right_knee = np.load('ik_right_knee.npy') * 0.20
ik_right_ankle = np.load('ik_right_ankle.npy') * 0.20

mj.mj_resetData(model, data)

# Interpolators
interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')

# Joint IDs - including root_x for translation
root_x_id = model.joint("root_x").id
leg_joint_ids = [model.joint(name).id for name in ['hip_link_2_1', 'link_2_1_link_1_1', 'link_1_1_foot_1',
                                                     'hip_link_2_2', 'link_2_2_link_1_2', 'link_1_2_foot_2']]

Kp = 600.0
Kd = 60.0

dt = model.opt.timestep
total_time = 50.0
time_dilation = 10.0
gait_period = ik_times[-1]

# Walking parameters
root_velocity_target = 0.35  # m/s - push the root forward
step_length_per_cycle = root_velocity_target * gait_period  # 0.35 * 3 = 1.05 m per cycle

print(f"[+] Root velocity target: {root_velocity_target:.2f} m/s")
print(f"[+] Expected step length: {step_length_per_cycle:.3f}m per {gait_period:.1f}s cycle")
print(f"[+] Leg control: Kp={Kp}, Kd={Kd}")
print(f"[+] Starting at X=0.150m (sand beginning)")
print()

t = 0.0
last_print = 0.0

while t < total_time:
    cycle_time = (t / gait_period) % 1.0
    traj_t = cycle_time * gait_period
    
    # Get IK targets
    q_l_hip = interp_left_hip(traj_t)
    q_l_knee = interp_left_knee(traj_t)
    q_l_ankle = interp_left_ankle(traj_t)
    q_r_hip = interp_right_hip(traj_t)
    q_r_knee = interp_right_knee(traj_t)
    q_r_ankle = interp_right_ankle(traj_t)
    
    # Control root_x with constant velocity - using forces directly, not actuators
    root_x_target = 0.150 + root_velocity_target * t / time_dilation
    hip_id = model.body("hip").id
    
    # Leg control
    ctrl = np.zeros(model.nu)
    
    targets = [q_l_hip, q_l_knee, q_l_ankle, q_r_hip, q_r_knee, q_r_ankle]
    for i, jid in enumerate(leg_joint_ids):
        q = data.qpos[jid]
        dq = data.qvel[jid]
        error = targets[i] - q
        ctrl[i] = Kp * error - Kd * dq
    
    # Apply forward force to move root
    root_x_error = root_x_target - data.qpos[root_x_id]
    forward_force = 1500.0 + 800.0 * root_x_error  # Strong constant push + proportional
    data.xfrc_applied[hip_id, 0] = forward_force
    
    data.ctrl[:] = ctrl
    mj.mj_step(model, data)
    t += dt * time_dilation
    
    # Print
    if t - last_print >= 1.0:
        x_pos = data.body("hip").xpos[0]
        dist = x_pos - 0.150
        
        try:
            l_z = data.body("foot_1").xpos[2]
            r_z = data.body("foot_2").xpos[2]
            l_s = "ON" if l_z < 0.455 else "UP"
            r_s = "ON" if r_z < 0.455 else "UP"
        except:
            l_s = r_s = "?"
        
        print(f"T: {t:6.1f}s | X: {x_pos:.4f}m | Dist: {dist:+.4f}m | L: {l_s} | R: {r_s} | Gait: {cycle_time*100:5.1f}%")
        last_print = t

print(f"\n[+] Final distance: {data.body('hip').xpos[0] - 0.150:.4f}m")
print(f"[+] Average speed: {(data.body('hip').xpos[0] - 0.150) / (t/time_dilation):.3f} m/s")
