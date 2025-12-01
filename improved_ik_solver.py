"""
Improved IK solver with better initialization strategies.
Uses previous solution as warm start + random seeds.
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
print("IMPROVED IK SOLVER - WITH WARM START")
print("=" * 80)

def get_feet_positions(data, model):
    foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot1_site")
    foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot2_site")
    return data.site_xpos[foot1_id].copy(), data.site_xpos[foot2_id].copy()

def solve_ik(target_foot1, target_foot2, initial_guess=None, num_seeds=20):
    """
    Solve IK using warm start (previous solution) + random seeds.
    """
    
    def objective(angles):
        data.qpos[3:9] = angles
        mujoco.mj_forward(model, data)
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
    
    # Try warm start first (if provided)
    if initial_guess is not None:
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8})
        error = objective(result.x)
        if error < best_error:
            best_error = error
            best_result = result.x
    
    # Try multiple random seeds
    for seed_idx in range(num_seeds):
        np.random.seed(seed_idx)
        x0 = np.random.uniform(-0.5, 0.5, 6)
        
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8})
        error = objective(result.x)
        if error < best_error:
            best_error = error
            best_result = result.x
    
    return best_result, best_error

print(f"\nComputing IK solutions for {len(hip_traj)} trajectory points...")
print(f"Using warm start + 20 random seeds per point\n")

num_steps = len(hip_traj)
joint_targets = np.zeros((num_steps, 6))
tracking_errors = np.zeros(num_steps)

start_time = time.time()
last_solution = None

for step in range(num_steps):
    # Get IK solution (warm start with previous solution)
    angles, error = solve_ik(foot1_traj[step], foot2_traj[step], initial_guess=last_solution, num_seeds=20)
    
    joint_targets[step] = angles
    last_solution = angles.copy()
    
    # Compute actual error
    data.qpos[3:9] = angles
    mujoco.mj_forward(model, data)
    f1, f2 = get_feet_positions(data, model)
    
    tracking_errors[step] = (
        np.linalg.norm(f1 - foot1_traj[step]) + 
        np.linalg.norm(f2 - foot2_traj[step])
    )
    
    if (step + 1) % 50 == 0:
        elapsed = time.time() - start_time
        rate = (step + 1) / elapsed
        eta = (num_steps - step) / rate if rate > 0 else 0
        print(f"Step {step+1:3d}/{num_steps} ({rate:5.2f} steps/s, ETA {eta:6.1f}s)")
        print(f"  Current error: {tracking_errors[step]*1000:.2f}mm")
        print(f"  Mean error so far: {np.mean(tracking_errors[:step+1])*1000:.2f}mm")

elapsed = time.time() - start_time
print(f"\n\nCompleted in {elapsed:.1f}s")

print(f"\n{'='*80}")
print("IK SOLVER RESULTS - WITH WARM START")
print(f"{'='*80}\n")

mean_error = np.mean(tracking_errors)
max_error = np.max(tracking_errors)

print(f"Mean tracking error:  {mean_error*1000:.2f}mm")
print(f"Max tracking error:   {max_error*1000:.2f}mm")
print(f"Std dev:              {np.std(tracking_errors)*1000:.2f}mm")
print(f"Median:               {np.median(tracking_errors)*1000:.2f}mm")
print(f"95th percentile:      {np.percentile(tracking_errors, 95)*1000:.2f}mm")

print(f"\nError distribution:")
print(f"  Errors > 10mm:        {np.sum(tracking_errors > 0.010)} points ({100*np.sum(tracking_errors > 0.010)/num_steps:.1f}%)")
print(f"  Errors > 5mm:         {np.sum(tracking_errors > 0.005)} points ({100*np.sum(tracking_errors > 0.005)/num_steps:.1f}%)")
print(f"  Errors < 5mm:         {np.sum(tracking_errors <= 0.005)} points ({100*np.sum(tracking_errors <= 0.005)/num_steps:.1f}%)")
print(f"  Errors < 1mm:         {np.sum(tracking_errors < 0.001)} points ({100*np.sum(tracking_errors < 0.001)/num_steps:.1f}%)")

# Save solutions
np.save("joint_targets_warmstart.npy", joint_targets)
np.save("tracking_errors_warmstart.npy", tracking_errors)

print(f"\nSolutions saved to:")
print(f"  joint_targets_warmstart.npy")
print(f"  tracking_errors_warmstart.npy")

if mean_error < 0.005:
    print(f"\n✓ SUCCESS: Mean error < 5mm! Solutions are suitable for walking.")
elif mean_error < 0.01:
    print(f"\n~ ACCEPTABLE: Mean error < 10mm. Should work for slow walking.")
else:
    print(f"\n! WARNING: Mean error > 10mm. May result in poor walking performance.")
