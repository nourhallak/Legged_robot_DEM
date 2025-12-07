"""
Detailed diagnostic of foot contact with actual contact forces
"""
import mujoco as mj
import numpy as np
from scipy.interpolate import interp1d

# Load sand model
model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("\n" + "=" * 90)
print("DETAILED FOOT CONTACT DIAGNOSTIC")
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
foot1_id = model.body("foot_1").id
foot2_id = model.body("foot_2").id

Kp = 600.0
Kd = 60.0
gait_period = ik_times[-1]
forward_force = 0.0

print(f"Sand surface Z: ~0.450m")
print(f"Hip height: {data.body('hip').xpos[2]:.4f}m\n")

# Run simulation and check contact at key points
test_times = [0, 6.25, 12.5, 18.75, 25.0]  # Points in gait cycle

for test_t in test_times:
    # Reset and set pose for this time
    mj.mj_resetData(model, data)
    data.qpos[model.joint("hip_link_2_1").id] = ik_left_hip[0]
    data.qpos[model.joint("link_2_1_link_1_1").id] = ik_left_knee[0]
    data.qpos[model.joint("link_1_1_foot_1").id] = ik_left_ankle[0]
    data.qpos[model.joint("hip_link_2_2").id] = ik_right_hip[0]
    data.qpos[model.joint("link_2_2_link_1_2").id] = ik_right_knee[0]
    data.qpos[model.joint("link_1_2_foot_2").id] = ik_right_ankle[0]
    mj.mj_forward(model, data)
    
    # Simulate to this time point
    t = 0.0
    while t < test_t:
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
        t += model.opt.timestep
    
    # Analyze contact
    l_z = data.body("foot_1").xpos[2]
    r_z = data.body("foot_2").xpos[2]
    
    l_knee = data.qpos[model.joint("link_2_1_link_1_1").id]
    r_knee = data.qpos[model.joint("link_2_2_link_1_2").id]
    
    l_ankle = data.qpos[model.joint("link_1_1_foot_1").id]
    r_ankle = data.qpos[model.joint("link_1_2_foot_2").id]
    
    print(f"Time {test_t:5.2f}s ({(test_t/gait_period)*100:5.1f}% of gait):")
    print(f"  Left foot:  Z={l_z:.6f}m (sand at 0.450m) | Knee={l_knee:+.3f}rad | Ankle={l_ankle:+.3f}rad")
    print(f"  Right foot: Z={r_z:.6f}m (sand at 0.450m) | Knee={r_knee:+.3f}rad | Ankle={r_ankle:+.3f}rad")
    
    # Count contacts
    nc = data.ncon
    l_contacts = 0
    r_contacts = 0
    for c_idx in range(nc):
        contact = data.contact[c_idx]
        if contact.geom1 in [model.geom(f"foot_1_geom").id, model.geom(f"foot_1_geom").id]:
            l_contacts += 1
        if contact.geom2 in [model.geom(f"foot_2_geom").id, model.geom(f"foot_2_geom").id]:
            r_contacts += 1
    
    print(f"  Contact count: Left={l_contacts}, Right={r_contacts}")
    print(f"  Total contacts: {nc}\n")

print("[+] Diagnostic complete")
