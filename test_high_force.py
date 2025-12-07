"""
Simple test: Robot on sand with VERY high force to overcome friction
"""
import mujoco as mj
import numpy as np
from scipy.interpolate import interp1d

# Load model
model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

# Load IK trajectories
ik_times = np.load('ik_times.npy')
ik_left_hip = np.load('ik_left_hip.npy') * 0.20
ik_left_knee = np.load('ik_left_knee.npy') * 0.20
ik_left_ankle = np.load('ik_left_ankle.npy') * 0.20
ik_right_hip = np.load('ik_right_hip.npy') * 0.20
ik_right_knee = np.load('ik_right_knee.npy') * 0.20
ik_right_ankle = np.load('ik_right_ankle.npy') * 0.20

mj.mj_resetData(model, data)

# Position robot
data.qpos[model.joint("root_x").id] = 0.150
data.qpos[model.joint("root_y").id] = 0.0

# Standing pose
data.qpos[model.joint("hip_link_2_1").id] = 0.0
data.qpos[model.joint("link_2_1_link_1_1").id] = -0.5
data.qpos[model.joint("link_1_1_foot_1").id] = 0.1

data.qpos[model.joint("hip_link_2_2").id] = 0.0
data.qpos[model.joint("link_2_2_link_1_2").id] = -0.5
data.qpos[model.joint("link_1_2_foot_2").id] = 0.1

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
forward_force = 3000.0

print("=" * 80)
print("TEST: ROBOT ON SAND WITH 3000N FORCE (NO VIEWER)")
print("=" * 80)
print(f"Initial X: {data.body('hip').xpos[0]:.4f}m")
print(f"Forward force: {forward_force}N\n")

start_x = data.body("hip").xpos[0]
t = 0.0

# Simulate for 30 seconds
max_time = 30.0
last_print = 0.0
positions = []

while t < max_time:
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
    
    # Apply 3000N forward force
    data.xfrc_applied[hip_id, 0] = forward_force
    
    data.ctrl[:] = ctrl
    mj.mj_step(model, data)
    t += model.opt.timestep
    
    # Print every second
    if t - last_print >= 1.0:
        x = data.body("hip").xpos[0]
        y = data.body("hip").xpos[1]
        dist = x - start_x
        positions.append(x)
        print(f"t={t:6.2f}s | X={x:.4f}m (+{dist:+.4f}m) | Y={y:+.5f}m | Velocity_X={data.qvel[model.joint('root_x').id]:+.4f}m/s")
        last_print = t

print(f"\nFinal X: {data.body('hip').xpos[0]:.4f}m")
print(f"Total distance: {data.body('hip').xpos[0] - start_x:+.4f}m")
print(f"Final velocity: {data.qvel[model.joint('root_x').id]:+.4f}m/s")

if len(positions) > 2:
    dists = [positions[i] - positions[0] for i in range(len(positions))]
    print(f"\nMovement:")
    print(f"  Started: {positions[0]:.4f}m")
    print(f"  Ended:   {positions[-1]:.4f}m")
    print(f"  Total:   {positions[-1] - positions[0]:+.4f}m")
