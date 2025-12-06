#!/usr/bin/env python3
"""
Improved Inverse Kinematics Solver

Considers base position + leg IK for better trajectory following.
"""

import numpy as np
from scipy.optimize import minimize
import mujoco

class ImprovedIKSolver:
    """IK solver that jointly optimizes base and leg poses."""
    
    def __init__(self, model_path):
        """Initialize solver."""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Joint indices
        self.base_x = 0
        self.base_y = 1
        self.base_rz = 2
        self.left_joints = [3, 4, 5]
        self.right_joints = [6, 7, 8]
        
        # Sites
        self.left_site = 0
        self.right_site = 1
    
    def solve_leg_ik(self, target_pos, joints, site_idx, base_pos, q_init=None):
        """Solve IK for a single leg given base position."""
        
        if q_init is None:
            q_init = np.array([0.0, -np.pi/4, np.pi/4])
        
        def objective(q_leg):
            """Minimize position error."""
            # Set base
            self.data.qpos[0:3] = base_pos
            # Set leg joints
            self.data.qpos[joints] = q_leg
            
            mujoco.mj_forward(self.model, self.data)
            
            ee_pos = self.data.site_xpos[site_idx]
            error = np.linalg.norm(ee_pos - target_pos)
            return error
        
        bounds = [(-np.pi, np.pi)] * 3
        result = minimize(
            objective,
            q_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        return result.x, result.fun
    
    def solve_trajectory(self, base_traj, foot1_traj, foot2_traj):
        """Solve IK for full trajectory."""
        
        n_steps = len(base_traj)
        q_left_traj = np.zeros((n_steps, 3))
        q_right_traj = np.zeros((n_steps, 3))
        error_left = np.zeros(n_steps)
        error_right = np.zeros(n_steps)
        
        print(f"\nSolving IK trajectory ({n_steps} steps)...")
        print(f"  Considering base position + leg IK")
        
        for step in range(n_steps):
            # Base position
            base_pos = np.array([base_traj[step, 0], base_traj[step, 1], 0.0])
            
            # Get previous solutions for warm start
            q_left_prev = q_left_traj[step-1] if step > 0 else None
            q_right_prev = q_right_traj[step-1] if step > 0 else None
            
            # Solve left leg
            q_left, err_left = self.solve_leg_ik(
                foot1_traj[step],
                self.left_joints,
                self.left_site,
                base_pos,
                q_init=q_left_prev
            )
            q_left_traj[step] = q_left
            error_left[step] = err_left
            
            # Solve right leg
            q_right, err_right = self.solve_leg_ik(
                foot2_traj[step],
                self.right_joints,
                self.right_site,
                base_pos,
                q_init=q_right_prev
            )
            q_right_traj[step] = q_right
            error_right[step] = err_right
            
            if (step + 1) % 50 == 0:
                mean_err = (error_left[:step+1].mean() + error_right[:step+1].mean()) / 2
                max_err = max(error_left[:step+1].max(), error_right[:step+1].max())
                print(f"  Step {step+1}/{n_steps}: Mean error: {mean_err*1000:.3f}mm, Max: {max_err*1000:.3f}mm")
        
        mean_err_left = error_left.mean()
        mean_err_right = error_right.mean()
        max_err_left = error_left.max()
        max_err_right = error_right.max()
        
        print(f"\nIK Solution Summary:")
        print(f"  Left leg:  Mean error: {mean_err_left*1000:.3f}mm, Max: {max_err_left*1000:.3f}mm")
        print(f"  Right leg: Mean error: {mean_err_right*1000:.3f}mm, Max: {max_err_right*1000:.3f}mm")
        
        return q_left_traj, q_right_traj, error_left, error_right


def main():
    """Generate IK solutions with improved method."""
    
    print("\n" + "="*80)
    print("IMPROVED INVERSE KINEMATICS SOLVER")
    print("="*80)
    
    # Load trajectories
    print("\nLoading trajectories...")
    base_traj = np.load("base_trajectory.npy")
    foot1_traj = np.load("foot1_trajectory.npy")
    foot2_traj = np.load("foot2_trajectory.npy")
    print(f"  Loaded {len(base_traj)} steps")
    
    # Initialize solver
    solver = ImprovedIKSolver("legged_robot_ik.xml")
    
    # Solve trajectory
    print("\n" + "="*80)
    print("SOLVING TRAJECTORY")
    print("="*80)
    
    q_left, q_right, err_left, err_right = solver.solve_trajectory(
        base_traj, foot1_traj, foot2_traj
    )
    
    # Save
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    np.save("left_leg_angles_improved.npy", q_left)
    np.save("right_leg_angles_improved.npy", q_right)
    np.save("left_leg_errors_improved.npy", err_left)
    np.save("right_leg_errors_improved.npy", err_right)
    
    print(f"✓ left_leg_angles_improved.npy")
    print(f"✓ right_leg_angles_improved.npy")
    print(f"✓ left_leg_errors_improved.npy")
    print(f"✓ right_leg_errors_improved.npy")
    
    # Generate report
    print("\n" + "="*80)
    print("JOINT ANGLE RANGES")
    print("="*80)
    
    print(f"\nLeft Leg:")
    for i in range(3):
        names = ["Hip", "Knee", "Ankle"]
        print(f"  {names[i]:8s}: {np.degrees(q_left[:, i].min()):7.1f}° to {np.degrees(q_left[:, i].max()):7.1f}°")
    
    print(f"\nRight Leg:")
    for i in range(3):
        names = ["Hip", "Knee", "Ankle"]
        print(f"  {names[i]:8s}: {np.degrees(q_right[:, i].min()):7.1f}° to {np.degrees(q_right[:, i].max()):7.1f}°")
    
    print("\n" + "="*80)
    print("IK SOLVING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
