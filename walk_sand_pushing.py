"""
Sand-pushing walking gait - Robot walks on sand by alternating stance and swing phases
Uses IK trajectories modulated by gait phase to create realistic walking
"""
import mujoco as mj
import numpy as np
from scipy.interpolate import interp1d

# Load model
model = mj.MjModel.from_xml_path('legged_robot_sand_shifted.xml')
data = mj.MjData(model)

print("=" * 80)
print("SAND-PUSHING WALKING - Realistic gait with alternating feet")
print("=" * 80)

# Load IK trajectories
ik_times = np.load('ik_times.npy')
ik_left_hip = np.load('ik_left_hip.npy')
ik_left_knee = np.load('ik_left_knee.npy')
ik_left_ankle = np.load('ik_left_ankle.npy')
ik_right_hip = np.load('ik_right_hip.npy')
ik_right_knee = np.load('ik_right_knee.npy')
ik_right_ankle = np.load('ik_right_ankle.npy')

# Reset to initial state
mj.mj_resetData(model, data)

# Scale trajectories
amplitude_scale = 0.20  # 20% amplitude for moderate stride
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
joint_ids = []
for name in joint_names:
    try:
        joint_ids.append(model.joint(name).id)
    except:
        pass

# Control parameters
Kp = 600.0
Kd = 60.0

# Get body ID for force
hip_id = model.body("hip").id

# Simulation parameters
dt = model.opt.timestep
total_time = 45.0  # 45 seconds of walking
time_dilation = 10.0  # 10x slower for visibility
gait_period = ik_times[-1]  # ~3 seconds per cycle

# Gait parameters
forward_force = 900.0  # Forward push force
stance_duration = gait_period * 0.6  # 60% of cycle in stance (push)
swing_duration = gait_period * 0.4   # 40% in swing (lift)

print(f"[+] Gait cycle: {gait_period:.2f}s")
print(f"[+] Stance: {stance_duration:.2f}s, Swing: {swing_duration:.2f}s")
print(f"[+] Forward push: {forward_force}N")
print(f"[+] Control: Kp={Kp}, Kd={Kd}")
print(f"[+] Amplitude: {amplitude_scale*100:.0f}%")
print(f"[+] Robot starts at X=0.150m (sand beginning)")
print()

t = 0.0
last_print = 0.0
cycle_count = 0

while t < total_time:
    # Calculate which leg is in stance
    cycle_time = (t / gait_period) % 1.0
    
    # Left leg stance: 0.0-0.5, Right leg stance: 0.5-1.0
    left_is_stance = cycle_time < 0.5
    right_is_stance = cycle_time >= 0.5
    
    # Get IK trajectory points (use cycle_time to interpolate through trajectory)
    traj_t = cycle_time * gait_period
    
    # Left leg targets
    target_left_hip = interp_left_hip(traj_t)
    target_left_knee = interp_left_knee(traj_t)
    target_left_ankle = interp_left_ankle(traj_t)
    
    # Right leg targets
    target_right_hip = interp_right_hip(traj_t)
    target_right_knee = interp_right_knee(traj_t)
    target_right_ankle = interp_right_ankle(traj_t)
    
    # Modulate amplitudes based on stance/swing
    # During stance: keep compressed, during swing: allow full trajectory
    if left_is_stance:
        # Left leg pushes - reduce hip movement, compress knee
        target_left_hip *= 0.4  # Reduce hip contribution
        target_left_knee = min(target_left_knee, -0.5)  # Keep compressed
    else:
        # Left leg swings - use full trajectory
        pass
    
    if right_is_stance:
        # Right leg pushes - reduce hip movement, compress knee
        target_right_hip *= 0.4  # Reduce hip contribution
        target_right_knee = min(target_right_knee, -0.5)  # Keep compressed
    else:
        # Right leg swings - use full trajectory
        pass
    
    # Joint control
    ctrl = np.zeros(model.nu)
    
    # Left leg (joints 0, 1, 2)
    for i in range(3):
        jid = joint_ids[i]
        target = [target_left_hip, target_left_knee, target_left_ankle][i]
        q = data.qpos[jid]
        dq = data.qvel[jid]
        error = target - q
        ctrl[i] = Kp * error - Kd * dq
    
    # Right leg (joints 3, 4, 5)
    for i in range(3):
        jid = joint_ids[3 + i]
        target = [target_right_hip, target_right_knee, target_right_ankle][i]
        q = data.qpos[jid]
        dq = data.qvel[jid]
        error = target - q
        ctrl[3 + i] = Kp * error - Kd * dq
    
    # Apply forward force (continuous for now)
    data.xfrc_applied[hip_id, 0] = forward_force
    
    # Set control
    data.ctrl[:] = ctrl
    
    # Step simulation
    mj.mj_step(model, data)
    t += dt * time_dilation
    
    # Print status
    if t - last_print >= 1.0:
        x_pos = data.body("hip").xpos[0]
        dist = x_pos - 0.150
        phase_pct = cycle_time * 100
        leg = "LEFT" if left_is_stance else "RIGHT"
        
        try:
            l_foot_z = data.body("foot_1").xpos[2]
            r_foot_z = data.body("foot_2").xpos[2]
            l_status = "ON" if l_foot_z < 0.455 else "UP"
            r_status = "ON" if r_foot_z < 0.455 else "UP"
        except:
            l_foot_z = r_foot_z = 0.0
            l_status = r_status = "?"
        
        print(f"T: {t:6.1f}s | Gait: {phase_pct:5.1f}% ({leg:5s}) | X: {x_pos:.4f}m | Dist: {dist:+.4f}m | L: {l_status} | R: {r_status}")
        last_print = t

print("\n[+] Simulation complete")
print(f"[+] Final position: X = {data.body('hip').xpos[0]:.4f}m")
print(f"[+] Distance traveled from sand start: {data.body('hip').xpos[0] - 0.150:.4f}m")
