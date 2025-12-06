#!/usr/bin/env python3
import numpy as np
import mujoco

xml_path = "legged_robot_ik.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

base_trajectory = np.load("base_trajectory.npy")
foot1_trajectory = np.load("foot1_trajectory.npy")
foot2_trajectory = np.load("foot2_trajectory.npy")
joint_targets = np.load("joint_targets_warmstart.npy")

num_steps = base_trajectory.shape[0]

print("=" * 80)
print("WALKING - EXPLICIT FORWARD VELOCITY")
print("=" * 80)

KP_legs = 20.0
KD_legs = 2.0
forward_velocity = 0.01  # 10mm/s
dt = model.opt.timestep

hip_positions = []
foot1_errors = []

foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot1_site")
foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot2_site")

data.qpos[:] = 0
mujoco.mj_forward(model, data)
data.qpos[3:9] = joint_targets[0, :]
mujoco.mj_forward(model, data)

last_angles = data.qpos[3:9].copy()
root_x_joint_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_x")
# For velocity, we use jnt_dofadr instead
vel_addr = model.jnt_dofadr[root_x_joint_idx] if root_x_joint_idx >= 0 else -1

max_sim_steps = num_steps * 5
sim_step_count = 0
frame_count = 0

while sim_step_count < max_sim_steps:
    traj_idx = sim_step_count % num_steps
    
    target_angles = joint_targets[traj_idx, :].copy()
    current_angles = data.qpos[3:9].copy()
    
    angle_errors = target_angles - current_angles
    angle_velocities = (current_angles - last_angles) / dt
    velocity_errors = -angle_velocities
    
    control_legs = KP_legs * angle_errors + KD_legs * velocity_errors
    data.ctrl[:] = np.clip(control_legs, -1.0, 1.0)
    
    if vel_addr >= 0:
        data.qvel[vel_addr] = forward_velocity
    
    mujoco.mj_step(model, data)
    
    if vel_addr >= 0:
        data.qvel[vel_addr] = forward_velocity
    
    last_angles = current_angles.copy()
    sim_step_count += 1
    frame_count += 1
    
    if frame_count % int(num_steps / 50) == 0:
        hip_pos = data.xpos[model.body("hip").id].copy()
        f1_pos = data.site_xpos[foot1_id].copy()
        f1_error = np.linalg.norm(f1_pos - foot1_trajectory[traj_idx, :])
        print(f"Step {sim_step_count:5d}: Foot1: {f1_error*1000:6.2f}mm, Hip X: {hip_pos[0]:8.6f}m")

hip_positions = np.array([data.xpos[model.body("hip").id].copy() for _ in range(1)])

print("\n" + "=" * 80)
print("FINAL HIP POSITION")
print("=" * 80)
hip_final = data.xpos[model.body("hip").id].copy()
print(f"Hip X: {hip_final[0]:.6f}m")
print("=" * 80)
