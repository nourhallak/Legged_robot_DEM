"""
Sand-pushing walking with COM translation
Robot walks by: 1) translating COM forward, 2) legs provide ground contact
"""
import mujoco as mj
import numpy as np
from scipy.interpolate import interp1d

# Load model
model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("=" * 80)
print("SAND WALKING WITH COM TRANSLATION")
print("=" * 80)

# Load IK trajectories
ik_times = np.load('ik_times.npy')
ik_left_hip = np.load('ik_left_hip.npy')
ik_left_knee = np.load('ik_left_knee.npy')
ik_left_ankle = np.load('ik_left_ankle.npy')
ik_right_hip = np.load('ik_right_hip.npy')
ik_right_knee = np.load('ik_right_knee.npy')
ik_right_ankle = np.load('ik_right_ankle.npy')

# Reset
mj.mj_resetData(model, data)

# Scale trajectories
amplitude_scale = 0.18  # 18% amplitude
ik_left_hip *= amplitude_scale
ik_left_knee *= amplitude_scale
ik_left_ankle *= amplitude_scale
ik_right_hip *= amplitude_scale
ik_right_knee *= amplitude_scale
ik_right_ankle *= amplitude_scale

# Create interpolators
interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')

# Joint IDs
joint_names = ['hip_link_2_1', 'link_2_1_link_1_1', 'link_1_1_foot_1',
               'hip_link_2_2', 'link_2_2_link_1_2', 'link_1_2_foot_2']
joint_ids = [model.joint(name).id for name in joint_names]

# Control parameters
Kp = 600.0
Kd = 60.0
hip_id = model.body("hip").id
gait_period = ik_times[-1]

# Simulation parameters
dt = model.opt.timestep
total_time = 40.0
time_dilation = 10.0

# Walking strategy: 
# - Alternate left/right stance
# - Push forward with stance leg
# - Swing non-stance leg
forward_push_base = 1200.0  # N - base forward force

print(f"[+] Gait period: {gait_period:.2f}s")
print(f"[+] Forward push: {forward_push_base}N")
print(f"[+] Control: Kp={Kp}, Kd={Kd}")
print(f"[+] Starting at X=0.150m (sand beginning)")
print()

t = 0.0
last_print = 0.0

while t < total_time:
    cycle_time = (t / gait_period) % 1.0
    
    # Left leg in stance for first half, right for second half
    left_stance = cycle_time < 0.5
    
    # Interpolate IK trajectory
    traj_t = cycle_time * gait_period
    
    # Get base trajectories
    q_left_hip = interp_left_hip(traj_t)
    q_left_knee = interp_left_knee(traj_t)
    q_left_ankle = interp_left_ankle(traj_t)
    q_right_hip = interp_right_hip(traj_t)
    q_right_knee = interp_right_knee(traj_t)
    q_right_ankle = interp_right_ankle(traj_t)
    
    # Modulate during stance: compress more to push into ground
    if left_stance:
        # Left pushes - enhance compression
        q_left_knee = min(q_left_knee - 0.2, -0.6)  # Compress more
        q_left_hip *= 0.3  # Reduce hip movement
        # Right swings
        q_right_hip *= 2.0  # Increase swing amplitude
        q_right_knee = max(q_right_knee, -0.3)  # Lift knee
    else:
        # Right pushes
        q_right_knee = min(q_right_knee - 0.2, -0.6)  # Compress more
        q_right_hip *= 0.3  # Reduce hip movement
        # Left swings
        q_left_hip *= 2.0  # Increase swing amplitude
        q_left_knee = max(q_left_knee, -0.3)  # Lift knee
    
    # Joint control
    ctrl = np.zeros(model.nu)
    targets = [q_left_hip, q_left_knee, q_left_ankle, q_right_hip, q_right_knee, q_right_ankle]
    
    for i, jid in enumerate(joint_ids):
        q = data.qpos[jid]
        dq = data.qvel[jid]
        error = targets[i] - q
        ctrl[i] = Kp * error - Kd * dq
    
    # Forward force - apply during entire cycle
    data.xfrc_applied[hip_id, 0] = forward_push_base
    
    data.ctrl[:] = ctrl
    mj.mj_step(model, data)
    t += dt * time_dilation
    
    # Print
    if t - last_print >= 1.0:
        x_pos = data.body("hip").xpos[0]
        dist = x_pos - 0.150
        phase = "LEFT" if left_stance else "RIGHT"
        
        try:
            l_z = data.body("foot_1").xpos[2]
            r_z = data.body("foot_2").xpos[2]
            l_s = "ON" if l_z < 0.455 else "UP"
            r_s = "ON" if r_z < 0.455 else "UP"
        except:
            l_s = r_s = "?"
        
        print(f"T: {t:6.1f}s | Phase: {phase:5s} ({cycle_time*100:5.1f}%) | X: {x_pos:.4f}m | Dist: {dist:+.4f}m | L: {l_s} | R: {r_s}")
        last_print = t

print(f"\n[+] Final distance: {data.body('hip').xpos[0] - 0.150:.4f}m")
