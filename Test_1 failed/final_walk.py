"""
Walking robot with interactive viewer that actively renders
"""
import numpy as np
import mujoco
import mujoco.viewer

def get_feet_positions(data, model):
    foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, 'foot1')
    foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, 'foot2')
    return data.xpos[foot1_id].copy(), data.xpos[foot2_id].copy()

def compute_ik(model, data, target_foot1, target_foot2, max_iterations=50):
    learning_rate = 0.2
    best_error = float('inf')
    best_qpos = data.qpos[6:12].copy()
    
    qpos_saved = data.qpos.copy()
    qvel_saved = data.qvel.copy()
    
    for iteration in range(max_iterations):
        foot1_pos, foot2_pos = get_feet_positions(data, model)
        err_foot1 = target_foot1 - foot1_pos
        err_foot2 = target_foot2 - foot2_pos
        total_error = np.linalg.norm(err_foot1) + np.linalg.norm(err_foot2)
        
        if total_error < best_error:
            best_error = total_error
            best_qpos = data.qpos[6:12].copy()
        
        if total_error < 5e-5:
            data.qpos[:] = qpos_saved
            data.qvel[:] = qvel_saved
            return data.qpos[6:12].copy(), total_error
        
        J = np.zeros((6, 6))
        dq = 1e-6
        for j in range(6):
            data.qpos[6 + j] += dq
            mujoco.mj_kinematics(model, data)
            f1p, f2p = get_feet_positions(data, model)
            J[0:3, j] = (f1p - foot1_pos) / dq
            J[3:6, j] = (f2p - foot2_pos) / dq
            data.qpos[6 + j] -= dq
        
        lambda_damp = 0.001 if iteration < 10 else 0.01
        JtJ_damped = J.T @ J + lambda_damp * np.eye(6)
        try:
            dq_sol = np.linalg.solve(JtJ_damped, J.T @ np.concatenate([err_foot1, err_foot2]))
        except:
            break
        
        joint_limits = np.array([[-1.57, 1.57], [-2.0944, 1.0472], [-1.57, 1.57],
                                 [-1.57, 1.57], [-2.0944, 1.0472], [-1.57, 1.57]])
        new_joints = data.qpos[6:12] + learning_rate * dq_sol
        new_joints = np.clip(new_joints, joint_limits[:, 0], joint_limits[:, 1])
        
        data.qpos[6:12] = new_joints
        mujoco.mj_kinematics(model, data)
    
    data.qpos[:] = qpos_saved
    data.qvel[:] = qvel_saved
    return best_qpos, best_error

print("\n" + "="*80)
print("ROBOT WALKING - INTERACTIVE VIEWER")
print("="*80)

model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

# Set slower timestep for better visibility
model.opt.timestep = 0.002

hip_traj = np.load('hip_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

num_steps = len(hip_traj)

print(f"Loaded {num_steps} trajectory steps")
print(f"Timestep: {model.opt.timestep*1000:.1f}ms")
print(f"\nStarting viewer - you should see the robot walking now!\n")

# Use launch_passive but with explicit rendering loop
viewer = mujoco.viewer.launch_passive(model, data)
viewer.cam.azimuth = 0
viewer.cam.elevation = -20
viewer.cam.lookat = np.array([0.005, 0.0, 0.22])
viewer.cam.distance = 0.5

step_idx = 0
frame = 0
num_cycles = 2

print(f"Running {num_cycles} cycles ({num_cycles * num_steps} trajectory steps)...\n")
print("Step | Hip_Z (m) | F1_Z (m) | Joint 1 | Joint 2 | Joint 3 | Joint 4 | Joint 5 | Joint 6")
print("-" * 95)

try:
    with viewer:
        while viewer.is_running() and step_idx < num_steps * num_cycles:
            current_step = step_idx % num_steps
            
            # Get targets
            target_hip = hip_traj[current_step]
            target_foot1 = foot1_traj[current_step]
            target_foot2 = foot2_traj[current_step]
            
            # Set base position
            data.qpos[0:3] = target_hip
            data.qpos[3:7] = [1, 0, 0, 0]
            
            # Compute IK
            target_joints, ik_error = compute_ik(model, data, target_foot1, target_foot2, max_iterations=50)
            
            # Apply control
            kp = 25
            kd = 5
            for i in range(6):
                joint_idx = 6 + i
                error = target_joints[i] - data.qpos[joint_idx]
                vel_error = 0 - data.qvel[joint_idx]
                data.ctrl[i] = kp * error + kd * vel_error
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Print every 100 steps
            if current_step % 100 == 0:
                f1, f2 = get_feet_positions(data, model)
                joint_deg = np.degrees(data.qpos[6:12])
                joint_str = " ".join([f"{j:7.1f}" for j in joint_deg])
                print(f"{current_step:4d} | {target_hip[2]:.4f}   | {f1[2]:.4f}  | {joint_str}")
            
            step_idx += 1
            frame += 1

except KeyboardInterrupt:
    print("\n\nSimulation interrupted by user")
