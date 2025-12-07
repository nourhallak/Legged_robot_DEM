"""
Simplified contact check - monitor foot heights during simulation
"""
import mujoco as mj
import numpy as np
from scipy.interpolate import interp1d

# Load sand model
model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("\n" + "=" * 90)
print("CONTACT DIAGNOSIS - CHECKING FOOT HEIGHTS AND MOTION")
print("=" * 90)

# Load IK trajectories
ik_times = np.load('ik_times.npy')
ik_left_hip = np.load('ik_left_hip.npy')
ik_left_knee = np.load('ik_left_knee.npy')
ik_left_ankle = np.load('ik_left_ankle.npy')
ik_right_hip = np.load('ik_right_hip.npy')
ik_right_knee = np.load('ik_right_knee.npy')
ik_right_ankle = np.load('ik_right_ankle.npy')

mj.mj_resetData(model, data)

# Set initial pose
data.qpos[model.joint("hip_link_2_1").id] = ik_left_hip[0]
data.qpos[model.joint("link_2_1_link_1_1").id] = ik_left_knee[0]
data.qpos[model.joint("link_1_1_foot_1").id] = ik_left_ankle[0]

data.qpos[model.joint("hip_link_2_2").id] = ik_right_hip[0]
data.qpos[model.joint("link_2_2_link_1_2").id] = ik_right_knee[0]
data.qpos[model.joint("link_1_2_foot_2").id] = ik_right_ankle[0]

mj.mj_forward(model, data)

# Interpolators
interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')

leg_joint_ids = [model.joint(name).id for name in ['hip_link_2_1', 'link_2_1_link_1_1', 'link_1_1_foot_1',
                                                     'hip_link_2_2', 'link_2_2_link_1_2', 'link_1_2_foot_2']]
hip_id = model.body("hip").id

Kp = 600.0
Kd = 60.0
gait_period = ik_times[-1]
forward_force = 0.0

print(f"Gait period: {gait_period:.2f}s")
print(f"Sand surface Z: 0.450m")
print(f"Initial hip Z: {data.body('hip').xpos[2]:.4f}m\n")

# Simulate one full gait cycle and collect data
l_z_min = 1.0
l_z_max = 0.0
r_z_min = 1.0
r_z_max = 0.0

l_z_values = []
r_z_values = []
gait_percentages = []

t = 0.0
step_count = 0
while t < gait_period:
    cycle_time = (t / gait_period) % 1.0
    traj_t = cycle_time * gait_period
    
    targets = [
        interp_left_hip(traj_t),
        interp_left_knee(traj_t),
        interp_left_ankle(traj_t),
        interp_right_hip(traj_t),
        interp_right_knee(traj_t),
        interp_right_ankle(traj_t)
    ]
    
    ctrl = np.zeros(model.nu)
    for i, jid in enumerate(leg_joint_ids):
        q = data.qpos[jid]
        dq = data.qvel[jid]
        error = targets[i] - q
        ctrl[i] = Kp * error - Kd * dq
    
    data.ctrl[:] = ctrl
    data.xfrc_applied[hip_id, 0] = forward_force
    
    mj.mj_step(model, data)
    
    l_z = data.body("foot_1").xpos[2]
    r_z = data.body("foot_2").xpos[2]
    
    l_z_min = min(l_z_min, l_z)
    l_z_max = max(l_z_max, l_z)
    r_z_min = min(r_z_min, r_z)
    r_z_max = max(r_z_max, r_z)
    
    l_z_values.append(l_z)
    r_z_values.append(r_z)
    gait_percentages.append(cycle_time * 100)
    
    if step_count % 50 == 0:  # Print every 50 steps
        l_contact = "ON " if l_z <= 0.455 else "UP "
        r_contact = "ON " if r_z <= 0.455 else "UP "
        print(f"Time {t:6.3f}s ({cycle_time*100:5.1f}%): L_Z={l_z:.6f}m ({l_contact}) | R_Z={r_z:.6f}m ({r_contact})")
    
    t += model.opt.timestep
    step_count += 1

print(f"\nFoot Z-height ranges during gait cycle:")
print(f"  Left foot:  {l_z_min:.6f}m to {l_z_max:.6f}m (range: {l_z_max-l_z_min:.6f}m)")
print(f"  Right foot: {r_z_min:.6f}m to {r_z_max:.6f}m (range: {r_z_max-r_z_min:.6f}m)")
print(f"  Sand surface: 0.450000m")
print(f"\nContact threshold: Z <= 0.455m")

# Check if any foot goes below threshold
l_stuck = l_z_min < 0.445  # Much below sand surface
r_stuck = r_z_min < 0.445
l_off = l_z_max > 0.460   # Well above sand
r_off = r_z_max > 0.460

print(f"\nDiagnosis:")
if l_stuck:
    print(f"  ⚠ LEFT FOOT appears to SINK below sand (Z={l_z_min:.6f}m < 0.445m)")
if r_stuck:
    print(f"  ⚠ RIGHT FOOT appears to SINK below sand (Z={r_z_min:.6f}m < 0.445m)")
if l_off:
    print(f"  ⚠ LEFT FOOT lifts HIGH above sand (Z={l_z_max:.6f}m > 0.460m)")
if r_off:
    print(f"  ⚠ RIGHT FOOT lifts HIGH above sand (Z={r_z_max:.6f}m > 0.460m)")

if not (l_stuck or r_stuck or l_off or r_off):
    print(f"  ✓ Both feet maintain good contact with sand (range ~0.450-0.455m)")

print(f"\n[+] Diagnostic complete")
