"""
CORE SOLUTION: Proper stepping gait that advances forward
No sand, just regular floor - get this working first
"""
import mujoco as mj
import mujoco.viewer
import numpy as np
from scipy.interpolate import interp1d

# Load MODEL WITHOUT SAND - use the original robot model
model = mj.MjModel.from_xml_path('legged_robot_ik.xml')
data = mj.MjData(model)

print("\n" + "=" * 90)
print(" " * 20 + "ROBOT WALKING ON REGULAR FLOOR")
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

# Position robot
data.qpos[model.joint("root_x").id] = 0.0
data.qpos[model.joint("root_y").id] = 0.0

# Standing pose
data.qpos[model.joint("hip_link_2_1").id] = ik_left_hip[0]
data.qpos[model.joint("link_2_1_link_1_1").id] = ik_left_knee[0]
data.qpos[model.joint("link_1_1_foot_1").id] = ik_left_ankle[0]

data.qpos[model.joint("hip_link_2_2").id] = ik_right_hip[0]
data.qpos[model.joint("link_2_2_link_1_2").id] = ik_right_knee[0]
data.qpos[model.joint("link_1_2_foot_2").id] = ik_right_ankle[0]

mj.mj_forward(model, data)

print(f"[+] Robot on regular floor")
print(f"[+] Initial X: {data.body('hip').xpos[0]:.4f}m, Y: {data.body('hip').xpos[1]:.4f}m")
print(f"[+] Left foot Z: {data.body('foot_1').xpos[2]:.4f}m")
print(f"[+] Right foot Z: {data.body('foot_2').xpos[2]:.4f}m")

# Interpolators
interp_left_hip = interp1d(ik_times, ik_left_hip, kind='cubic', fill_value='extrapolate')
interp_left_knee = interp1d(ik_times, ik_left_knee, kind='cubic', fill_value='extrapolate')
interp_left_ankle = interp1d(ik_times, ik_left_ankle, kind='cubic', fill_value='extrapolate')
interp_right_hip = interp1d(ik_times, ik_right_hip, kind='cubic', fill_value='extrapolate')
interp_right_knee = interp1d(ik_times, ik_right_knee, kind='cubic', fill_value='extrapolate')
interp_right_ankle = interp1d(ik_times, ik_right_ankle, kind='cubic', fill_value='extrapolate')

leg_joint_ids = [model.joint(name).id for name in ['hip_link_2_1', 'link_2_1_link_1_1', 'link_1_1_foot_1',
                                                     'hip_link_2_2', 'link_2_2_link_1_2', 'link_1_2_foot_2']]

Kp = 300.0  # Lower gain for stability
Kd = 30.0
gait_period = ik_times[-1]

print(f"[+] Control: Kp={Kp}, Kd={Kd}")
print(f"[+] Gait period: {gait_period:.2f}s\n")
print("[+] Starting viewer - watch robot walk forward on floor...\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 1.5
    viewer.cam.elevation = -25
    viewer.cam.azimuth = 90
    
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
        mj.mj_step(model, data)
        viewer.sync()
        t += model.opt.timestep
        
        if t - last_print >= 1.0:
            x = data.body("hip").xpos[0]
            y = data.body("hip").xpos[1]
            dist = x - start_x
            
            l_z = data.body("foot_1").xpos[2]
            r_z = data.body("foot_2").xpos[2]
            
            print(f"t={t:6.2f}s | X={x:.4f}m (+{dist:+.4f}m) | Y={y:+.6f}m | L-Z={l_z:.4f} R-Z={r_z:.4f}")
            last_print = t

print("\n[+] Viewer closed")
