"""
Advanced IK solver with multi-start L-BFGS-B + SLSQP for aggressive error reduction.
Uses 50 random seeds and tries both methods to find best solution.
"""

import numpy as np
import mujoco
from scipy.optimize import minimize
import time

xml_path = "legged_robot_ik.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Load trajectories
hip_traj = np.load("base_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

print("=" * 80)
print("AGGRESSIVE MULTI-START IK SOLVER")
print("=" * 80)

def get_feet_positions(data, model):
    """Get current foot positions from forward kinematics."""
    foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot1_site")
    foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot2_site")
    return data.site_xpos[foot1_id].copy(), data.site_xpos[foot2_id].copy()

def set_joint_angles(data, model, angles):
    """Set leg joint angles (indices 3-8, base is 0-2)."""
    qpos_idx = [3, 4, 5, 6, 7, 8]
    for idx, angle in zip(qpos_idx, angles):
        data.qpos[idx] = angle
    mujoco.mj_forward(model, data)

def ik_solver_aggressive(target_foot1, target_foot2, num_seeds=50):
    """
    Solve IK using multiple random seeds with both L-BFGS-B and SLSQP.
    Returns the best solution found.
    """
    
    def objective(angles):
        set_joint_angles(data, model, angles)
        f1, f2 = get_feet_positions(data, model)
        error_f1 = np.sum((f1 - target_foot1)**2)
        error_f2 = np.sum((f2 - target_foot2)**2)
        return error_f1 + error_f2
    
    bounds = [
        (-1.57, 1.57), (-2.0944, 1.0472), (-1.57, 1.57),
        (-1.57, 1.57), (-2.0944, 1.0472), (-1.57, 1.57),
    ]
    
    best_result = None
    best_error = float('inf')
    
    # Try multiple random seeds with L-BFGS-B (faster)
    for seed_idx in range(num_seeds):
        np.random.seed(seed_idx)
        x0 = np.random.uniform(-0.5, 0.5, 6)
        
        result_lbfgs = minimize(
            objective, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-7, 'gtol': 1e-6}
        )
        
        error = objective(result_lbfgs.x)
        if error < best_error:
            best_error = error
            best_result = result_lbfgs.x
        
        # Every 10 seeds, also try SLSQP on the best candidate so far
        if seed_idx % 10 == 0 and seed_idx > 0:
            result_slsqp = minimize(
                objective, best_result, method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-7}
            )
            error_slsqp = objective(result_slsqp.x)
            if error_slsqp < best_error:
                best_error = error_slsqp
                best_result = result_slsqp.x
    
    # Final polish with SLSQP
    result_final = minimize(
        objective, best_result, method='SLSQP',
        bounds=bounds,
        options={'maxiter': 300, 'ftol': 1e-8}
    )
    error_final = objective(result_final.x)
    if error_final < best_error:
        best_result = result_final.x
    
    return best_result

# ============ COMPUTE OPTIMIZED IK SOLUTIONS ============
print(f"\nComputing optimized IK solutions for {len(hip_traj)} trajectory points...")
print(f"Using 50 random seeds + SLSQP refinement per point\n")

num_steps = len(hip_traj)
joint_targets = np.zeros((num_steps, 6))
tracking_errors = np.zeros(num_steps)

start_time = time.time()

for step in range(num_steps):
    if step % 20 == 0:
        elapsed = time.time() - start_time
        rate = (step + 1) / elapsed if elapsed > 0 else 0
        eta = (num_steps - step) / rate if rate > 0 else 0
        print(f"  Step {step:3d}/{num_steps} ({rate:5.2f} steps/s, ETA {eta:6.1f}s)...", end='\r')
    
    # Solve IK with aggressive multi-start
    angles = ik_solver_aggressive(foot1_traj[step], foot2_traj[step], num_seeds=50)
    joint_targets[step] = angles
    
    # Compute error
    set_joint_angles(data, model, angles)
    f1, f2 = get_feet_positions(data, model)
    tracking_errors[step] = (
        np.linalg.norm(f1 - foot1_traj[step]) + 
        np.linalg.norm(f2 - foot2_traj[step])
    )

elapsed = time.time() - start_time
print(f"\n\nCompleted in {elapsed:.1f}s")

print(f"\n{'='*80}")
print("OPTIMIZED IK RESULTS")
print(f"{'='*80}\n")
print(f"Mean tracking error:  {np.mean(tracking_errors)*1000:.2f}mm")
print(f"Max tracking error:   {np.max(tracking_errors)*1000:.2f}mm  (TARGET: < 5mm)")
print(f"Std dev:              {np.std(tracking_errors)*1000:.2f}mm")
print(f"Median:               {np.median(tracking_errors)*1000:.2f}mm")
print(f"95th percentile:      {np.percentile(tracking_errors, 95)*1000:.2f}mm")
print(f"\nErrors > 10mm:        {np.sum(tracking_errors > 0.010)} points")
print(f"Errors > 5mm:         {np.sum(tracking_errors > 0.005)} points")
print(f"Errors < 1mm:         {np.sum(tracking_errors < 0.001)} points")

np.save("joint_targets_aggressive.npy", joint_targets)
np.save("tracking_errors_aggressive.npy", tracking_errors)

print(f"\n{'='*80}")
