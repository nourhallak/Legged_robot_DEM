#!/usr/bin/env python3
"""
Test IK solution accuracy with uniform foot heights
Checks if feet are actually reaching their target positions during simulation
"""

import numpy as np
import mujoco as mj
import os

# Create a reference to the physics engine
xml_path = 'legged_robot_ik.xml'
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# Load trajectories
base_traj = np.load('base_trajectory.npy')
com_traj = np.load('com_trajectory.npy')
foot1_traj = np.load('foot1_trajectory.npy')
foot2_traj = np.load('foot2_trajectory.npy')

# IK solver implementation (same as ik_simulation.py)
def compute_ik_solution(initial_qpos, base_target, com_target, foot1_target, foot2_target, 
                        max_iterations=50, tolerance=0.002, learning_rate=0.08):
    """
    Solve IK for floating base (12 DOF) using numerical Jacobian
    """
    q = initial_qpos.copy()
    
    for iteration in range(max_iterations):
        # Current state kinematics
        data.qpos = q
        mj.mj_forward(model, data)
        
        # Get current end-effector positions
        base_pos = data.qpos[0:3]
        com_frame_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "com_frame")
        com_pos = data.xpos[com_frame_id]
        foot1_frame_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "foot1_frame")
        foot1_pos = data.xpos[foot1_frame_id]
        foot2_frame_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "foot2_frame")
        foot2_pos = data.xpos[foot2_frame_id]
        
        # Compute errors (unweighted, equal priority)
        base_error = base_target - base_pos
        com_error = com_target - com_pos
        foot1_error = foot1_target - foot1_pos
        foot2_error = foot2_target - foot2_pos
        
        # Stack errors as vector
        errors = np.concatenate([base_error, com_error, foot1_error, foot2_error])
        error_norm = np.linalg.norm(errors)
        
        # Check convergence
        if error_norm < tolerance:
            return q, iteration, error_norm, 0.0
        
        # Numerical Jacobian: compute how joint velocity affects each error
        # 12 errors × 9 variables (3 base + 6 joint angles)
        J = np.zeros((12, 9))
        epsilon = 0.001
        
        for j in range(9):
            q_perturbed = q.copy()
            q_perturbed[j] += epsilon
            
            data.qpos = q_perturbed
            mj.mj_forward(model, data)
            
            base_pos_p = data.qpos[0:3]
            com_pos_p = data.xpos[com_frame_id]
            foot1_pos_p = data.xpos[foot1_frame_id]
            foot2_pos_p = data.xpos[foot2_frame_id]
            
            errors_p = np.concatenate([
                base_target - base_pos_p,
                com_target - com_pos_p,
                foot1_target - foot1_pos_p,
                foot2_target - foot2_pos_p
            ])
            
            J[:, j] = (errors_p - errors) / epsilon
        
        # Update: q_new = q + alpha * J^T * errors
        dq = learning_rate * J.T @ errors
        q += dq
        
        # Enforce joint limits
        for j in range(3, 9):
            if j < 6:  # Hip and knee joints
                if j % 3 == 1:  # Knee joint
                    q[j] = np.clip(q[j], -2.0944, 1.0472)  # -120° to +60°
                else:  # Hip and ankle
                    q[j] = np.clip(q[j], -1.57, 1.57)  # ±90°
            else:  # Ankle
                q[j] = np.clip(q[j], -1.57, 1.57)
    
    return q, max_iterations, error_norm, 0.0

# Test initial IK
print("\n=== TESTING IK WITH UNIFORM 15MM FOOT HEIGHTS ===\n")

# Start with rest pose
q_rest = model.qpos0.copy()
q_current = q_rest.copy()

test_steps = [0, 50, 100, 150, 200, 250, 300, 350]
max_slip = 0.0
total_slip = 0.0

for step in test_steps:
    q_current, iters, error, _ = compute_ik_solution(
        q_current,
        base_traj[step],
        com_traj[step],
        foot1_traj[step],
        foot2_traj[step],
        max_iterations=50,
        tolerance=0.002,
        learning_rate=0.08
    )
    
    # Get actual positions after IK
    data.qpos = q_current
    mj.mj_forward(model, data)
    
    base_pos = data.qpos[0:3]
    com_frame_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "com_frame")
    com_pos = data.xpos[com_frame_id]
    foot1_frame_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "foot1_frame")
    foot1_pos = data.xpos[foot1_frame_id]
    foot2_frame_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "foot2_frame")
    foot2_pos = data.xpos[foot2_frame_id]
    
    # Compute actual errors
    foot1_slip_mm = (foot1_traj[step, 0] - foot1_pos[0]) * 1000
    foot2_slip_mm = (foot2_traj[step, 0] - foot2_pos[0]) * 1000
    foot1_z_error = (foot1_traj[step, 2] - foot1_pos[2]) * 1000
    foot2_z_error = (foot2_traj[step, 2] - foot2_pos[2]) * 1000
    
    slip = max(abs(foot1_slip_mm), abs(foot2_slip_mm))
    max_slip = max(max_slip, slip)
    total_slip += slip
    
    phase = "Stance" if (step % 200) < 100 else ("Swing" if (step % 200) < 100 else "Stance")
    
    print(f"Step {step:3d} ({phase}): Iters={iters}, Error={error:.6f}m")
    print(f"  Foot1 X slip: {foot1_slip_mm:7.2f}mm, Z error: {foot1_z_error:6.2f}mm")
    print(f"  Foot2 X slip: {foot2_slip_mm:7.2f}mm, Z error: {foot2_z_error:6.2f}mm")
    print()

print(f"\n=== RESULTS ===")
print(f"Max slip across test steps: {max_slip:.2f}mm")
print(f"Average slip: {total_slip/len(test_steps):.2f}mm")

if max_slip < 5.0:
    print("✓ EXCELLENT: Feet are reaching targets (< 5mm slip)")
elif max_slip < 10.0:
    print("✓ GOOD: Feet mostly reaching targets (< 10mm slip)")
elif max_slip < 20.0:
    print("⚠ FAIR: Some foot slipping but manageable (< 20mm slip)")
else:
    print("✗ POOR: Significant foot slipping (> 20mm)")
