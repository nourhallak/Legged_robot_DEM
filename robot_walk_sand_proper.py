"""
Robot walking on SAND with properly-designed forward gait
"""
import mujoco as mj
import mujoco.viewer
import numpy as np
from scipy.interpolate import interp1d

# Load sand model
model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("\n" + "=" * 90)
print(" " * 15 + "ROBOT WALKING ON SAND WITH PROPER GAIT")
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

# Position robot at sand start
data.qpos[model.joint("root_x").id] = 0.150
data.qpos[model.joint("root_y").id] = 0.0

# Set initial pose from trajectory
data.qpos[model.joint("hip_link_2_1").id] = ik_left_hip[0]
data.qpos[model.joint("link_2_1_link_1_1").id] = ik_left_knee[0]
data.qpos[model.joint("link_1_1_foot_1").id] = ik_left_ankle[0]

data.qpos[model.joint("hip_link_2_2").id] = ik_right_hip[0]
data.qpos[model.joint("link_2_2_link_1_2").id] = ik_right_knee[0]
data.qpos[model.joint("link_1_2_foot_2").id] = ik_right_ankle[0]

mj.mj_forward(model, data)

print(f"[+] Robot positioned at X={data.body('hip').xpos[0]:.4f}m (sand start), Y={data.body('hip').xpos[1]:.4f}m")
print(f"[+] Left foot Z: {data.body('foot_1').xpos[2]:.4f}m")
print(f"[+] Right foot Z: {data.body('foot_2').xpos[2]:.4f}m")
print(f"[+] Sand surface: Z~0.450m")

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
forward_force = 500.0  # Assist force to help stepping

print(f"[+] Control: Kp={Kp}, Kd={Kd}, Gait period={gait_period:.2f}s")
print(f"[+] Forward assist force: {forward_force}N")
print("[+] Starting viewer - robot walking on sand...\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.0
    viewer.cam.elevation = -25
    viewer.cam.azimuth = 90
    
    # Contact visualization
    viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    
    t = 0.0
    start_x = data.body("hip").xpos[0]
    last_print = 0.0
    
    while viewer.is_running():
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
        
        # Apply small forward assist force
        data.xfrc_applied[hip_id, 0] = forward_force
        
        mj.mj_step(model, data)
        viewer.sync()
        t += model.opt.timestep
        
        if t - last_print >= 1.0:
            x = data.body("hip").xpos[0]
            y = data.body("hip").xpos[1]
            dist = x - start_x
            
            l_z = data.body("foot_1").xpos[2]
            r_z = data.body("foot_2").xpos[2]
            l_contact = "ON " if l_z <= 0.455 else "UP "
            r_contact = "ON " if r_z <= 0.455 else "UP "
            
            print(f"t={t:6.1f}s | X={x:.4f}m (+{dist:+.4f}m) | Y={y:+.6f}m | "
                  f"L-foot:{l_contact} R-foot:{r_contact} | Cycle:{cycle_time*100:5.1f}%")
            
            last_print = t

print("\n[+] Simulation ended")
