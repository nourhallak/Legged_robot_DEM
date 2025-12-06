#!/usr/bin/env python3
"""
Improved IK Solver with Better Accuracy

Uses multiple optimization strategies to achieve <5mm error.
"""

import numpy as np
import mujoco
from scipy.optimize import minimize, least_squares
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("IMPROVED INVERSE KINEMATICS - TARGET <5MM ERROR")
print("="*80)

# Load model and data
model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
data = mujoco.MjData(model)

# Load trajectories
print("\nLoading trajectories...")
base_planned = np.load("base_feasible.npy")
foot1_planned = np.load("foot1_feasible.npy")
foot2_planned = np.load("foot2_feasible.npy")
q_left_old = np.load("q_left_feasible.npy")
q_right_old = np.load("q_right_feasible.npy")

NUM_STEPS = len(q_left_old)
CYCLE = 100
STANCE = 60
FLAT_FOOT_ANGLE = 0.0

# Parameters for improved IK
q_left = np.zeros((NUM_STEPS, 3))
q_right = np.zeros((NUM_STEPS, 3))
err_left = np.zeros(NUM_STEPS)
err_right = np.zeros(NUM_STEPS)

def solve_ik_improved(target_pos, step, foot_idx, leg='left'):
    """
    Improved IK solver with multiple optimization strategies.
    """
    cycle = step % CYCLE
    in_stance = cycle < STANCE
    
    def objective(q):
        """Objective function for optimization."""
        if in_stance:
            data.qpos[foot_idx:foot_idx+3] = [q[0], q[1], FLAT_FOOT_ANGLE]
        else:
            data.qpos[foot_idx:foot_idx+3] = q
        mujoco.mj_forward(model, data)
        site_idx = 0 if leg == 'left' else 1
        return data.site_xpos[site_idx] - target_pos
    
    def objective_norm(q):
        """Norm of objective for minimization."""
        return np.linalg.norm(objective(q))
    
    # Get initial guess from previous step
    if step > 0:
        if leg == 'left':
            q_init = q_left[step-1].copy()
        else:
            q_init = q_right[step-1].copy()
    else:
        q_init = np.array([0.0, -np.pi/4, np.pi/4])
    
    best_sol = None
    best_error = float('inf')
    
    # Strategy 1: L-BFGS-B with tight tolerance
    if in_stance:
        result1 = minimize(objective_norm, q_init[:2], method='L-BFGS-B',
                          bounds=[(-np.pi, np.pi), (-np.pi, np.pi)],
                          options={'ftol': 1e-8, 'maxiter': 300})
        if result1.fun < best_error:
            best_error = result1.fun
            sol = result1.x
            q_sol = np.array([sol[0], sol[1], FLAT_FOOT_ANGLE])
            best_sol = q_sol
    else:
        result1 = minimize(objective_norm, q_init, method='L-BFGS-B',
                          bounds=[(-np.pi, np.pi)]*3,
                          options={'ftol': 1e-8, 'maxiter': 300})
        if result1.fun < best_error:
            best_error = result1.fun
            best_sol = result1.x
    
    # Strategy 2: Least squares (for non-linear least squares)
    if in_stance:
        result2 = least_squares(objective, q_init[:2], 
                               bounds=([-np.pi, -np.pi], [np.pi, np.pi]),
                               ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=300)
        error2 = np.linalg.norm(result2.fun)
        if error2 < best_error:
            best_error = error2
            sol = result2.x
            q_sol = np.array([sol[0], sol[1], FLAT_FOOT_ANGLE])
            best_sol = q_sol
    else:
        result2 = least_squares(objective, q_init,
                               bounds=([-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi]),
                               ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=300)
        error2 = np.linalg.norm(result2.fun)
        if error2 < best_error:
            best_error = error2
            best_sol = result2.x
    
    # Strategy 3: Try multiple random initial guesses if error still high
    if best_error > 0.005:  # If error > 5mm, try random initializations
        for _ in range(5):
            q_rand = np.random.uniform(-np.pi, np.pi, 2 if in_stance else 3)
            if in_stance:
                result = minimize(objective_norm, q_rand, method='L-BFGS-B',
                                 bounds=[(-np.pi, np.pi), (-np.pi, np.pi)],
                                 options={'ftol': 1e-9, 'maxiter': 400})
                if result.fun < best_error:
                    best_error = result.fun
                    sol = result.x
                    q_sol = np.array([sol[0], sol[1], FLAT_FOOT_ANGLE])
                    best_sol = q_sol
            else:
                result = minimize(objective_norm, q_rand, method='L-BFGS-B',
                                 bounds=[(-np.pi, np.pi)]*3,
                                 options={'ftol': 1e-9, 'maxiter': 400})
                if result.fun < best_error:
                    best_error = result.fun
                    best_sol = result.x
    
    return best_sol, best_error

print(f"\nSolving IK for {NUM_STEPS} points (improved solver)...")
print("Using: L-BFGS-B, Least Squares, and multi-start optimization\n")

for step in range(NUM_STEPS):
    # Solve left foot
    q_left[step], err_left[step] = solve_ik_improved(foot1_planned[step], step, 3, 'left')
    
    # Solve right foot
    q_right[step], err_right[step] = solve_ik_improved(foot2_planned[step], step, 6, 'right')
    
    if (step + 1) % 50 == 0:
        print(f"  {step+1}/{NUM_STEPS}: L err {err_left[step]*1000:.3f}mm, R err {err_right[step]*1000:.3f}mm")

print(f"\n✓ IK Solving Complete\n")

# Statistics
print("LEFT FOOT RESULTS:")
print(f"  Mean error: {err_left.mean()*1000:.3f} mm")
print(f"  Max error:  {err_left.max()*1000:.3f} mm")
print(f"  < 5mm:      {np.sum(err_left < 0.005)}/{NUM_STEPS}")

print(f"\nRIGHT FOOT RESULTS:")
print(f"  Mean error: {err_right.mean()*1000:.3f} mm")
print(f"  Max error:  {err_right.max()*1000:.3f} mm")
print(f"  < 5mm:      {np.sum(err_right < 0.005)}/{NUM_STEPS}")

print(f"\nCOMBINED:")
print(f"  Both < 5mm: {np.sum((err_left < 0.005) & (err_right < 0.005))}/{NUM_STEPS}")

# Verify solutions
print("\nVerifying solutions with forward kinematics...")
foot1_actual = np.zeros((NUM_STEPS, 3))
foot2_actual = np.zeros((NUM_STEPS, 3))

for step in range(NUM_STEPS):
    data.qpos[3:6] = q_left[step]
    data.qpos[6:9] = q_right[step]
    mujoco.mj_forward(model, data)
    foot1_actual[step] = data.site_xpos[0].copy()
    foot2_actual[step] = data.site_xpos[1].copy()

err_foot1_verify = np.linalg.norm(foot1_actual - foot1_planned, axis=1)
err_foot2_verify = np.linalg.norm(foot2_actual - foot2_planned, axis=1)

print(f"\nVERIFICATION (actual FK):")
print(f"  Left foot mean:  {err_foot1_verify.mean()*1000:.3f} mm")
print(f"  Right foot mean: {err_foot2_verify.mean()*1000:.3f} mm")
print(f"  Left foot max:   {err_foot1_verify.max()*1000:.3f} mm")
print(f"  Right foot max:  {err_foot2_verify.max()*1000:.3f} mm")

# Save results
print("\nSaving improved solutions...")
np.save('q_left_improved.npy', q_left)
np.save('q_right_improved.npy', q_right)
np.save('err_left_improved.npy', err_left)
np.save('err_right_improved.npy', err_right)

# Check if acceptable
left_ok = np.sum(err_foot1_verify < 0.005) / NUM_STEPS * 100
right_ok = np.sum(err_foot2_verify < 0.005) / NUM_STEPS * 100

print(f"\nQUALITY ASSESSMENT:")
print(f"  Left foot <5mm:  {left_ok:.1f}%")
print(f"  Right foot <5mm: {right_ok:.1f}%")

if err_foot1_verify.mean() < 0.005 and err_foot2_verify.mean() < 0.005:
    print("\n✓✓✓ EXCELLENT - Both feet < 5mm mean error ✓✓✓")
elif err_foot1_verify.max() < 0.005 and err_foot2_verify.max() < 0.005:
    print("\n✓✓ GOOD - All errors < 5mm ✓✓")
elif left_ok > 80 and right_ok > 80:
    print("\n✓ ACCEPTABLE - Most errors < 5mm ✓")
else:
    print("\n⚠ NEEDS IMPROVEMENT - Too many errors > 5mm ⚠")

print("\n" + "="*80)
