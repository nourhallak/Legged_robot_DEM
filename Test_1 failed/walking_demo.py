"""
Simple walking demonstration with real-time IK solving.
"""
import numpy as np
import mujoco
import mujoco.viewer

def load_model_with_assets():
    """Load MuJoCo model with mesh assets"""
    return mujoco.MjModel.from_xml_path('legged_robot_ik.xml')

def get_feet_positions(data, model):
    """Get current feet positions"""
    foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, 'foot1')
    foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, 'foot2')
    
    foot1_pos = data.xpos[foot1_id].copy()
    foot2_pos = data.xpos[foot2_id].copy()
    
    return foot1_pos, foot2_pos

def compute_ik(model, data, target_foot1, target_foot2, max_iterations=50):
    """Solve IK for feet positions using damped least-squares"""
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
        
        # Compute Jacobian
        J = np.zeros((6, 6))
        dq = 1e-6
        for j in range(6):
            data.qpos[6 + j] += dq
            mujoco.mj_kinematics(model, data)
            f1p, f2p = get_feet_positions(data, model)
            J[0:3, j] = (f1p - foot1_pos) / dq
            J[3:6, j] = (f2p - foot2_pos) / dq
            data.qpos[6 + j] -= dq
        
        # Damped least-squares
        lambda_damp = 0.001 if iteration < 10 else 0.01
        JtJ_damped = J.T @ J + lambda_damp * np.eye(6)
        try:
            dq_sol = np.linalg.solve(JtJ_damped, J.T @ np.concatenate([err_foot1, err_foot2]))
        except:
            break
        
        # Update with joint limits
        joint_limits = np.array([[-1.57, 1.57], [-2.0944, 1.0472], [-1.57, 1.57],
                                 [-1.57, 1.57], [-2.0944, 1.0472], [-1.57, 1.57]])
        new_joints = data.qpos[6:12] + learning_rate * dq_sol
        new_joints = np.clip(new_joints, joint_limits[:, 0], joint_limits[:, 1])
        
        data.qpos[6:12] = new_joints
        mujoco.mj_kinematics(model, data)
    
    data.qpos[6:12] = best_qpos
    mujoco.mj_kinematics(model, data)
    return best_qpos, best_error

print("Loading model and trajectories...")
model = load_model_with_assets()
data = mujoco.MjData(model)

# Load trajectories
hip_traj = np.load('hip_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

num_steps = len(hip_traj)
print(f"Loaded {num_steps} steps")

# Create viewer
print("Starting visualization...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Configure camera
    viewer.cam.azimuth = 0
    viewer.cam.elevation = -20
    viewer.cam.lookat = np.array([0.005, 0.0, 0.22])
    viewer.cam.distance = 0.5
    
    step_idx = 0
    frame_count = 0
    
    while viewer.is_running():
        # Get trajectory targets
        target_hip = hip_traj[step_idx]
        target_foot1 = foot1_traj[step_idx]
        target_foot2 = foot2_traj[step_idx]
        
        # Set hip position (base)
        data.qpos[0:3] = target_hip
        data.qpos[3:7] = [1, 0, 0, 0]  # Identity quaternion
        
        # Solve IK for feet
        joints, ik_error = compute_ik(model, data, target_foot1, target_foot2, max_iterations=50)
        data.qpos[6:12] = joints
        
        # Forward kinematics and physics
        mujoco.mj_kinematics(model, data)
        mujoco.mj_step(model, data)
        
        # Print progress every 50 steps
        if step_idx % 50 == 0:
            f1_pos, f2_pos = get_feet_positions(data, model)
            print(f"Step {step_idx:3d}: Hip_Z={target_hip[2]:.4f}m, "
                  f"Foot1_Z={f1_pos[2]:.4f}m, IK_Error={ik_error*1000:.1f}mm")
        
        # Move to next step
        step_idx = (step_idx + 1) % num_steps
        frame_count += 1

print(f"\nSimulation complete! Rendered {frame_count} frames")
