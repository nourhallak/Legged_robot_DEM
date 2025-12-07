"""
Robot walking on sand - PROPER FORWARD MOTION
Using pure forward-stepping gait (NO lateral Y motion)
"""
import mujoco as mj
import mujoco.viewer
import numpy as np
from scipy.interpolate import interp1d

# Load model
model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("\n" + "=" * 90)
print(" " * 15 + "ROBOT WALKING ON SAND - FORWARD MOTION ONLY")
print("=" * 90)

# Load IK trajectories
ik_times = np.load('ik_times.npy')
ik_left_hip = np.load('ik_left_hip.npy') * 0.20
ik_left_knee = np.load('ik_left_knee.npy') * 0.20
ik_left_ankle = np.load('ik_left_ankle.npy') * 0.20
ik_right_hip = np.load('ik_right_hip.npy') * 0.20
ik_right_knee = np.load('ik_right_knee.npy') * 0.20
ik_right_ankle = np.load('ik_right_ankle.npy') * 0.20

# Reset
mj.mj_resetData(model, data)

# Position robot ON sand at beginning
data.qpos[model.joint("root_x").id] = 0.150  # Start at sand beginning
data.qpos[model.joint("root_y").id] = 0.0    # Center Y
data.qpos[model.joint("root_rz").id] = 0.0   # No rotation

# Set initial standing pose
data.qpos[model.joint("hip_link_2_1").id] = 0.0
data.qpos[model.joint("link_2_1_link_1_1").id] = -0.5
data.qpos[model.joint("link_1_1_foot_1").id] = 0.1

data.qpos[model.joint("hip_link_2_2").id] = 0.0
data.qpos[model.joint("link_2_2_link_1_2").id] = -0.5
data.qpos[model.joint("link_1_2_foot_2").id] = 0.1

mj.mj_forward(model, data)

print(f"[+] Robot positioned at X={data.body('hip').xpos[0]:.4f}m, Y={data.body('hip').xpos[1]:.4f}m")
print(f"[+] Left foot Z: {data.body('foot_1').xpos[2]:.4f}m")
print(f"[+] Right foot Z: {data.body('foot_2').xpos[2]:.4f}m")
print(f"[+] Sand surface: Z=0.442m to 0.450m")

# Create interpolators
interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')

# Joint IDs
leg_joint_ids = [model.joint(name).id for name in ['hip_link_2_1', 'link_2_1_link_1_1', 'link_1_1_foot_1',
                                                     'hip_link_2_2', 'link_2_2_link_1_2', 'link_1_2_foot_2']]
hip_id = model.body("hip").id

# Control parameters
Kp = 600.0
Kd = 60.0
gait_period = ik_times[-1]
forward_force = 3000.0  # Increased to overcome sand static friction

print(f"[+] Control: Kp={Kp}, Kd={Kd}, Forward Force={forward_force}N")
print(f"[+] Gait period: {gait_period:.2f}s")
print("\n[+] Starting simulation - robot walking forward on sand...")
print("[+] Watch: X increases (forward), Y stays ~0 (straight line)\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -25
    viewer.cam.azimuth = 90
    
    t = 0.0
    start_x = data.body("hip").xpos[0]
    last_print = 0.0
    
    while viewer.is_running():
        # Gait cycle
        cycle_time = (t / gait_period) % 1.0
        traj_t = cycle_time * gait_period
        
        # Get joint targets
        targets = [
            interp_left_hip(traj_t),
            interp_left_knee(traj_t),
            interp_left_ankle(traj_t),
            interp_right_hip(traj_t),
            interp_right_knee(traj_t),
            interp_right_ankle(traj_t)
        ]
        
        # Joint control
        ctrl = np.zeros(model.nu)
        for i, jid in enumerate(leg_joint_ids):
            q = data.qpos[jid]
            dq = data.qvel[jid]
            error = targets[i] - q
            ctrl[i] = Kp * error - Kd * dq
        
        # Apply forward force ONLY (no Y force for straight line)
        data.xfrc_applied[hip_id, 0] = forward_force
        data.xfrc_applied[hip_id, 1] = 0.0  # NO Y force
        data.xfrc_applied[hip_id, 2] = 0.0
        
        data.ctrl[:] = ctrl
        mj.mj_step(model, data)
        
        with viewer.lock():
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        
        viewer.sync()
        t += model.opt.timestep
        
        # Print status every 1 second
        if t - last_print >= 1.0:
            x_pos = data.body("hip").xpos[0]
            y_pos = data.body("hip").xpos[1]
            dist = x_pos - start_x
            
            l_z = data.body("foot_1").xpos[2]
            r_z = data.body("foot_2").xpos[2]
            l_contact = "YES" if l_z <= 0.455 else "NO "
            r_contact = "YES" if r_z <= 0.455 else "NO "
            
            step = int(cycle_time * 20)  # 20 steps per cycle for display
            
            print(f"t={t:7.2f}s | X={x_pos:.4f}m (+{dist:+.4f}m) | Y={y_pos:+.5f}m | "
                  f"L-foot:{l_contact} R-foot:{r_contact} | Step: {step:2d}/20")
            
            last_print = t

print("\n[+] Viewer closed - simulation ended")
