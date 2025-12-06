#!/usr/bin/env python3
"""
Trajectory-Based IK Solver

Converts end-effector trajectories to joint angles considering base movement.
Uses relative foot positions from the base.
"""

import numpy as np
from scipy.optimize import minimize
import mujoco

class TrajectoryIKSolver:
    """Solves IK for trajectories with moving base."""
    
    def __init__(self, model_path):
        """Initialize."""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.left_joints = [3, 4, 5]
        self.right_joints = [6, 7, 8]
        self.left_site = 0
        self.right_site = 1
    
    def solve_leg_with_base(self, target_pos, joints, site_idx, base_z, q_init=None):
        """Solve IK for leg assuming base at origin."""
        
        if q_init is None:
            # Good initial guess for biped walking
            q_init = np.array([0.0, -np.pi/4, np.pi/4])
        
        def objective(q_leg):
            """Minimize position error."""
            # Set base to origin with z height
            self.data.qpos[0:3] = [0.0, 0.0, 0.0]
            # Set leg
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
            options={'maxiter': 200, 'ftol': 1e-9}
        )
        
        return result.x, result.fun
    
    def solve_trajectory(self, base_traj, foot1_traj, foot2_traj):
        """Solve IK treating feet relative to hip."""
        
        n_steps = len(base_traj)
        q_left = np.zeros((n_steps, 3))
        q_right = np.zeros((n_steps, 3))
        err_left = np.zeros(n_steps)
        err_right = np.zeros(n_steps)
        
        print(f"\nSolving trajectory-based IK ({n_steps} steps)...")
        print(f"  Base from trajectory, feet relative to hip")
        
        for step in range(n_steps):
            # Hip position from trajectory
            hip_pos = base_traj[step]
            
            # Foot positions relative to hip
            foot1_rel = foot1_traj[step] - hip_pos
            foot2_rel = foot2_traj[step] - hip_pos
            
            # Warm start from previous solution
            q_left_init = q_left[step-1] if step > 0 else None
            q_right_init = q_right[step-1] if step > 0 else None
            
            # Solve for each leg
            q_l, err_l = self.solve_leg_with_base(
                foot1_rel, self.left_joints, self.left_site, hip_pos[2], q_left_init
            )
            q_left[step] = q_l
            err_left[step] = err_l
            
            q_r, err_r = self.solve_leg_with_base(
                foot2_rel, self.right_joints, self.right_site, hip_pos[2], q_right_init
            )
            q_right[step] = q_r
            err_right[step] = err_r
            
            if (step + 1) % 50 == 0:
                mean_err = (err_left[:step+1].mean() + err_right[:step+1].mean()) / 2
                success = np.sum(err_left[:step+1] < 0.01) + np.sum(err_right[:step+1] < 0.01)
                print(f"  Step {step+1}/{n_steps}: Mean error: {mean_err*1000:.4f}mm, "
                      f"Success: {success}/100")
        
        # Summary
        success_left = np.sum(err_left < 0.01)
        success_right = np.sum(err_right < 0.01)
        
        print(f"\nSolution Summary:")
        print(f"  Left:  {success_left}/{n_steps} converged, "
              f"Mean: {err_left.mean()*1000:.4f}mm, Max: {err_left.max()*1000:.4f}mm")
        print(f"  Right: {success_right}/{n_steps} converged, "
              f"Mean: {err_right.mean()*1000:.4f}mm, Max: {err_right.max()*1000:.4f}mm")
        
        return q_left, q_right, err_left, err_right


def main():
    """Run trajectory-based IK."""
    
    print("\n" + "="*80)
    print("TRAJECTORY-BASED INVERSE KINEMATICS")
    print("="*80)
    
    # Load
    print("\nLoading...")
    base = np.load("base_trajectory.npy")
    foot1 = np.load("foot1_trajectory.npy")
    foot2 = np.load("foot2_trajectory.npy")
    print(f"  {len(base)} trajectory points loaded")
    
    # Solve
    solver = TrajectoryIKSolver("legged_robot_ik.xml")
    
    print("\n" + "="*80)
    print("SOLVING INVERSE KINEMATICS")
    print("="*80)
    
    q_left, q_right, err_left, err_right = solver.solve_trajectory(base, foot1, foot2)
    
    # Save
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    np.save("q_left.npy", q_left)
    np.save("q_right.npy", q_right)
    np.save("err_left.npy", err_left)
    np.save("err_right.npy", err_right)
    
    print(f"✓ q_left.npy (shape: {q_left.shape})")
    print(f"✓ q_right.npy (shape: {q_right.shape})")
    
    # Joint ranges
    print("\n" + "="*80)
    print("JOINT ANGLE RANGES")
    print("="*80)
    
    joint_names = ["Hip", "Knee", "Ankle"]
    print(f"\nLeft Leg:")
    for i in range(3):
        print(f"  {joint_names[i]:8s}: {np.degrees(q_left[:, i].min()):7.1f}° to {np.degrees(q_left[:, i].max()):7.1f}°")
    
    print(f"\nRight Leg:")
    for i in range(3):
        print(f"  {joint_names[i]:8s}: {np.degrees(q_right[:, i].min()):7.1f}° to {np.degrees(q_right[:, i].max()):7.1f}°")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
