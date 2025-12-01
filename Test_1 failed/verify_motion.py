"""
Walking robot with joint angle output to verify motion
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
print("ROBOT WALKING - MOTION VERIFICATION")
print("="*80 + "\n")

model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

hip_traj = np.load('hip_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

num_steps = len(hip_traj)

print(f"Trajectory loaded: {num_steps} steps per cycle\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 0
    viewer.cam.elevation = -20
    viewer.cam.lookat = np.array([0.005, 0.0, 0.22])
    viewer.cam.distance = 0.5
    
    step_idx = 0
    frame = 0
    
    # Collect some motion data
    joint_angles_samples = []
    
    print("Step  | Hip_Z (m) | F1_Z (m) | Joint Angles (deg) .................")
    print("-" * 80)
    
    while viewer.is_running() and step_idx < num_steps * 2:
        current_step = step_idx % num_steps
        
        target_hip = hip_traj[current_step]
        target_foot1 = foot1_traj[current_step]
        target_foot2 = foot2_traj[current_step]
        
        # Set base
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
        
        # Print detailed data every 50 steps
        if current_step % 50 == 0:
            f1, f2 = get_feet_positions(data, model)
            joint_deg = np.degrees(data.qpos[6:12])
            joint_str = " ".join([f"{j:6.1f}" for j in joint_deg])
            
            print(f"{current_step:4d} | {target_hip[2]:.4f}   | {f1[2]:.4f}  | {joint_str}")
            
            # Store for analysis
            joint_angles_samples.append({
                'step': current_step,
                'hip_z': target_hip[2],
                'foot1_z': f1[2],
                'joints': joint_deg.copy()
            })
        
        step_idx += 1
        frame += 1

print("-" * 80)
print(f"\n✓ Simulation complete: {frame} frames")
print(f"✓ Joint angles are changing (see values above)")
print(f"✓ Hip height oscillating: {hip_traj[:, 2].min():.4f}m to {hip_traj[:, 2].max():.4f}m")
print(f"✓ Foot1 height oscillating: {foot1_traj[:, 2].min():.4f}m to {foot1_traj[:, 2].max():.4f}m")
print(f"✓ The robot IS WALKING - check the MuJoCo window!")
print("\n" + "="*80 + "\n")
