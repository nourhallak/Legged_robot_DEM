"""
Robot walking on sand - proper positioning and straight-line motion
"""
import mujoco as mj
import mujoco.viewer
import numpy as np
from scipy.interpolate import interp1d

# Load model
model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("=" * 90)
print(" " * 20 + "ROBOT WALKING ON SAND - PROPER SETUP")
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

# Position robot ON sand at beginning (X=0.150, Y=0, Z adjusted to stand on sand)
# Sand surface is at Z=0.442m, robot hip needs to be higher for standing
data.qpos[model.joint("root_x").id] = 0.150  # Start at sand beginning
data.qpos[model.joint("root_y").id] = 0.0    # Center Y
data.qpos[model.joint("root_rz").id] = 0.0   # No rotation

# Set initial standing pose
data.qpos[model.joint("hip_link_2_1").id] = 0.0    # L-hip
data.qpos[model.joint("link_2_1_link_1_1").id] = -0.5  # L-knee bent
data.qpos[model.joint("link_1_1_foot_1").id] = 0.5    # L-ankle

data.qpos[model.joint("hip_link_2_2").id] = 0.0    # R-hip
data.qpos[model.joint("link_2_2_link_1_2").id] = -0.5  # R-knee bent
data.qpos[model.joint("link_1_2_foot_2").id] = 0.5    # R-ankle

mj.mj_forward(model, data)

print("[+] Robot positioned on sand at X=0.150m")
print("[+] Initial foot heights:")
print(f"    L-foot Z: {data.body('foot_1').xpos[2]:.4f}m")
print(f"    R-foot Z: {data.body('foot_2').xpos[2]:.4f}m")

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
root_x_id = model.joint("root_x").id

# Control parameters
Kp = 600.0
Kd = 60.0
gait_period = ik_times[-1]

print("[+] Starting viewer with robot walking on sand...")
print("[+] Watch the robot walk in a STRAIGHT LINE with both feet ON sand")
print("[+] Close the viewer window to exit\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set camera to follow and show sand clearly
    viewer.cam.distance = 2.0
    viewer.cam.elevation = -30
    viewer.cam.azimuth = 90
    
    t = 0.0
    last_print = 0.0
    
    while viewer.is_running():
        # Gait cycle
        cycle_time = (t / gait_period) % 1.0
        traj_t = cycle_time * gait_period
        
        # Get joint targets
        q_l_hip = interp_left_hip(traj_t)
        q_l_knee = interp_left_knee(traj_t)
        q_l_ankle = interp_left_ankle(traj_t)
        q_r_hip = interp_right_hip(traj_t)
        q_r_knee = interp_right_knee(traj_t)
        q_r_ankle = interp_right_ankle(traj_t)
        
        # Joint control
        ctrl = np.zeros(model.nu)
        targets = [q_l_hip, q_l_knee, q_l_ankle, q_r_hip, q_r_knee, q_r_ankle]
        
        for i, jid in enumerate(leg_joint_ids):
            q = data.qpos[jid]
            dq = data.qvel[jid]
            error = targets[i] - q
            ctrl[i] = Kp * error - Kd * dq
        
        # Push forward with consistent force (straight line motion)
        data.xfrc_applied[hip_id, 0] = 1500.0  # Forward X force
        data.xfrc_applied[hip_id, 1] = 0.0     # NO Y force - keep straight!
        
        data.ctrl[:] = ctrl
        mj.mj_step(model, data)
        
        # Update viewer with contact visualization
        with viewer.lock():
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        
        viewer.sync()
        t += model.opt.timestep
        
        # Print status
        if t - last_print >= 1.0:
            x_pos = data.body("hip").xpos[0]
            y_pos = data.body("hip").xpos[1]
            dist = x_pos - 0.150
            
            try:
                l_z = data.body("foot_1").xpos[2]
                r_z = data.body("foot_2").xpos[2]
                l_contact = "ON " if l_z < 0.455 else "UP "
                r_contact = "ON " if r_z < 0.455 else "UP "
            except:
                l_contact = r_contact = "? "
            
            gait_pct = cycle_time * 100
            print(f"t={t:6.1f}s | X={x_pos:.4f}m Y={y_pos:+.4f}m | Dist={dist:+.4f}m | L:{l_contact} R:{r_contact} | Gait:{gait_pct:5.1f}%")
            last_print = t

print("\n[+] Simulation ended")
