#!/usr/bin/env python3
"""
Inverse Kinematics Solver for Biped Robot (Simplified)

Direct implementation for the specific robot model.
"""

import numpy as np
from scipy.optimize import minimize
import mujoco
from pathlib import Path

class SimpleBipedIKSolver:
    """Simple IK solver for the biped robot."""
    
    def __init__(self, model_path):
        """Initialize with model."""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Joint indices for control
        # Joints: 0=root_x, 1=root_y, 2=root_rz, 3=hip1, 4=knee1, 5=ankle1, 6=hip2, 7=knee2, 8=ankle2
        self.left_joints = [3, 4, 5]      # hip_link_2_1, link_2_1_link_1_1, link_1_1_foot_1
        self.right_joints = [6, 7, 8]     # hip_link_2_2, link_2_2_link_1_2, link_1_2_foot_2
        
        # Site indices
        self.left_site = 0                # foot1_site
        self.right_site = 1               # foot2_site
    
    def forward_kinematics(self, q_left, q_right):
        """Compute foot positions."""
        # Set joint angles
        self.data.qpos[self.left_joints] = q_left
        self.data.qpos[self.right_joints] = q_right
        
        mujoco.mj_forward(self.model, self.data)
        
        left_pos = self.data.site_xpos[self.left_site].copy()
        right_pos = self.data.site_xpos[self.right_site].copy()
        
        return left_pos, right_pos
    
    def inverse_kinematics(self, target_pos, joints, site_idx, q_init=None, max_iter=100):
        """Solve IK for one foot."""
        if q_init is None:
            q_init = np.zeros(3)
        
        def objective(q):
            """Error function."""
            self.data.qpos[joints] = q
            mujoco.mj_forward(self.model, self.data)
            
            ee_pos = self.data.site_xpos[site_idx]
            error = np.linalg.norm(ee_pos - target_pos)
            return error
        
        # Optimize
        bounds = [(-np.pi, np.pi)] * 3
        result = minimize(
            objective,
            q_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-6}
        )
        
        return result.x, result.fun < 1e-3, result.fun
    
    def solve_trajectory(self, foot_trajectory, joints, site_idx, q_init=None):
        """Solve IK for trajectory."""
        n_points = len(foot_trajectory)
        q_trajectory = np.zeros((n_points, 3))
        success_mask = np.zeros(n_points, dtype=bool)
        error_array = np.zeros(n_points)
        
        print(f"\nSolving IK ({n_points} points)...")
        
        for i, target in enumerate(foot_trajectory):
            # Warm start with previous solution
            q_guess = q_trajectory[i-1] if i > 0 else (q_init[i] if q_init is not None else None)
            
            q_sol, success, error = self.inverse_kinematics(
                target, joints, site_idx, q_init=q_guess, max_iter=50
            )
            
            q_trajectory[i] = q_sol
            success_mask[i] = success
            error_array[i] = error
            
            if (i + 1) % 50 == 0:
                rate = np.sum(success_mask[:i+1]) / (i + 1) * 100
                print(f"  {i+1}/{n_points}: {rate:.1f}% success, max error: {error_array[:i+1].max()*1000:.2f}mm")
        
        rate = np.sum(success_mask) / n_points * 100
        print(f"  Final: {rate:.1f}% success, mean error: {error_array.mean()*1000:.2f}mm")
        
        return q_trajectory, success_mask, error_array


def main():
    """Generate IK solutions."""
    
    print("\n" + "="*80)
    print("BIPED ROBOT IK SOLVER (SIMPLIFIED)")
    print("="*80)
    
    # Load trajectories
    print("\nLoading trajectories...")
    base_traj = np.load("base_trajectory.npy")
    foot1_traj = np.load("foot1_trajectory.npy")
    foot2_traj = np.load("foot2_trajectory.npy")
    print(f"  Loaded {len(base_traj)} trajectory points")
    
    # Initialize solver
    print("\nInitializing solver...")
    model_path = "legged_robot_ik.xml"
    if not Path(model_path).exists():
        print(f"ERROR: {model_path} not found")
        return
    
    solver = SimpleBipedIKSolver(model_path)
    print(f"✓ Model loaded")
    
    # Solve IK
    print("\n" + "="*80)
    print("SOLVING INVERSE KINEMATICS")
    print("="*80)
    
    q_left, left_success, left_errors = solver.solve_trajectory(
        foot1_traj, solver.left_joints, solver.left_site
    )
    
    q_right, right_success, right_errors = solver.solve_trajectory(
        foot2_traj, solver.right_joints, solver.right_site
    )
    
    # Save
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    np.save("left_leg_angles.npy", q_left)
    np.save("right_leg_angles.npy", q_right)
    np.save("left_leg_success.npy", left_success)
    np.save("right_leg_success.npy", right_success)
    np.save("left_leg_errors.npy", left_errors)
    np.save("right_leg_errors.npy", right_errors)
    
    print(f"✓ left_leg_angles.npy   {q_left.shape}")
    print(f"✓ right_leg_angles.npy  {q_right.shape}")
    print(f"✓ left_leg_errors.npy   (mean: {left_errors.mean()*1000:.2f}mm, max: {left_errors.max()*1000:.2f}mm)")
    print(f"✓ right_leg_errors.npy  (mean: {right_errors.mean()*1000:.2f}mm, max: {right_errors.max()*1000:.2f}mm)")
    
    print("\n" + "="*80)
    print("IK SOLVING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
