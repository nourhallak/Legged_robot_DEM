"""
Walking demonstration - with keyboard control and window focus
"""
import numpy as np
import mujoco
import mujoco.viewer

def get_feet_positions(data, model):
    foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, 'foot1')
    foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, 'foot2')
    foot1_pos = data.xpos[foot1_id].copy()
    foot2_pos = data.xpos[foot2_id].copy()
    return foot1_pos, foot2_pos

def compute_ik(model, data, target_foot1, target_foot2, max_iterations=50):
    learning_rate = 0.2
    best_error = float('inf')
    best_qpos = data.qpos[6:12].copy()
    
    for iteration in range(max_iterations):
        foot1_pos, foot2_pos = get_feet_positions(data, model)
        err_foot1 = target_foot1 - foot1_pos
        err_foot2 = target_foot2 - foot2_pos
        total_error = np.linalg.norm(err_foot1) + np.linalg.norm(err_foot2)
        
        if total_error < best_error:
            best_error = total_error
            best_qpos = data.qpos[6:12].copy()
        
        if total_error < 5e-5:
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
    
    data.qpos[6:12] = best_qpos
    mujoco.mj_kinematics(model, data)
    return best_qpos, best_error

print("=" * 60)
print("WALKING ROBOT SIMULATOR")
print("=" * 60)
print("\nLoading model and trajectories...")

model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

hip_traj = np.load('hip_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

num_steps = len(hip_traj)

print(f"Loaded {num_steps} trajectory steps")
print("\nOpening MuJoCo viewer...")
print("The robot should be walking in the window below.")
print("Watch the console for step updates.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 0
    viewer.cam.elevation = -20
    viewer.cam.lookat = np.array([0.005, 0.0, 0.22])
    viewer.cam.distance = 0.5
    
    step_idx = 0
    frame = 0
    
    # Run for 10 cycles
    total_steps = num_steps * 10
    
    while viewer.is_running() and step_idx < total_steps:
        current_step = step_idx % num_steps
        
        target_hip = hip_traj[current_step]
        target_foot1 = foot1_traj[current_step]
        target_foot2 = foot2_traj[current_step]
        
        # Set base
        data.qpos[0:3] = target_hip
        data.qpos[3:7] = [1, 0, 0, 0]
        
        # Solve IK
        joints, error = compute_ik(model, data, target_foot1, target_foot2, max_iterations=50)
        data.qpos[6:12] = joints
        
        # Physics step
        mujoco.mj_kinematics(model, data)
        mujoco.mj_step(model, data)
        
        # Print updates
        if current_step % 50 == 0:
            f1, f2 = get_feet_positions(data, model)
            cycle = step_idx // num_steps
            print(f"Cycle {cycle}: Step {current_step:3d} | "
                  f"Hip_Z={target_hip[2]:.4f}m | "
                  f"Foot1_Z={f1[2]:.4f}m | "
                  f"Foot2_Z={f2[2]:.4f}m | "
                  f"Error={error*1000:.1f}mm")
        
        step_idx += 1
        frame += 1

print(f"\n{'='*60}")
print(f"Simulation complete: {frame} frames rendered")
print(f"{'='*60}")
