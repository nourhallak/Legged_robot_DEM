#!/usr/bin/env python3
"""
Inverse Kinematics Solver for Biped Robot

Solves IK for a biped robot with 3 DOF per leg (hip, knee, ankle).
Converts foot end-effector trajectories to joint angle commands.
"""

import numpy as np
from scipy.optimize import minimize
import mujoco
from pathlib import Path

class BipedIKSolver:
    """IK solver for biped robot with 2 legs."""
    
    def __init__(self, model_path):
        """
        Initialize IK solver with MuJoCo model.
        
        Args:
            model_path: Path to URDF/XML model file
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Find leg joint indices
        self.leg_joints = {}
        joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        for i, name in enumerate(joint_names):
            if 'left' in name.lower():
                self.leg_joints.setdefault('left', []).append(i)
            elif 'right' in name.lower():
                self.leg_joints.setdefault('right', []).append(i)
        
        # Find foot site indices
        self.foot_sites = {}
        site_names = [self.model.site(i).name for i in range(self.model.nsite)]
        for i, name in enumerate(site_names):
            if 'left_foot' in name.lower() or 'left' in name.lower():
                self.foot_sites['left'] = i
            elif 'right_foot' in name.lower() or 'right' in name.lower():
                self.foot_sites['right'] = i
    
    def forward_kinematics(self, q_left, q_right):
        """
        Compute forward kinematics.
        
        Args:
            q_left: Left leg joint angles (3,)
            q_right: Right leg joint angles (3,)
            
        Returns:
            left_pos, right_pos: End-effector positions (3,) each
        """
        # Get first available left and right leg sites
        left_site = self.foot_sites.get('left', 0)
        right_site = self.foot_sites.get('right', 1)
        
        # Set joint angles (adjust based on actual model structure)
        if len(q_left) > 0:
            self.data.qpos[:len(q_left)] = q_left
        if len(q_right) > 0 and len(q_left) + len(q_right) <= len(self.data.qpos):
            self.data.qpos[len(q_left):len(q_left)+len(q_right)] = q_right
        
        mujoco.mj_forward(self.model, self.data)
        
        left_pos = self.data.site_xpos[left_site].copy()
        right_pos = self.data.site_xpos[right_site].copy() if len(self.foot_sites) > 1 else left_pos.copy()
        
        return left_pos, right_pos
    
    def inverse_kinematics(self, target_pos, side='left', q_init=None, max_iter=100):
        """
        Solve IK using numerical optimization.
        
        Args:
            target_pos: Target end-effector position (3,)
            side: 'left' or 'right'
            q_init: Initial joint angles (3,). If None, uses zeros.
            max_iter: Maximum optimization iterations
            
        Returns:
            q_solution: Joint angles (3,), or None if IK failed
            success: Boolean success flag
            error: Position error at solution
        """
        if q_init is None:
            q_init = np.zeros(3)
        
        site_idx = self.foot_sites.get(side, 0)
        
        def objective(q):
            """Minimize position error."""
            self.data.qpos[:3] = q
            mujoco.mj_forward(self.model, self.data)
            
            ee_pos = self.data.site_xpos[site_idx]
            error = np.linalg.norm(ee_pos - target_pos)
            return error
        
        # Solve with bounds on joint angles
        bounds = [(-np.pi, np.pi)] * 3  # ±180° per joint
        
        result = minimize(
            objective,
            q_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-6}
        )
        
        q_solution = result.x
        success = result.fun < 1e-3  # Success if error < 1mm
        error = result.fun
        
        return q_solution, success, error
    
    def solve_trajectory(self, foot_trajectory, side='left', q_init=None):
        """
        Solve IK for entire trajectory.
        
        Args:
            foot_trajectory: Array of target positions (N, 3)
            side: 'left' or 'right'
            q_init: Optional array of initial joint angles (N, 3)
            
        Returns:
            q_trajectory: Joint angle trajectory (N, 3)
            success_mask: Boolean array indicating IK success for each point
            error_array: Position errors for each point
        """
        n_points = len(foot_trajectory)
        q_trajectory = np.zeros((n_points, 3))
        success_mask = np.zeros(n_points, dtype=bool)
        error_array = np.zeros(n_points)
        
        print(f"\nSolving IK for {side} foot ({n_points} points)...")
        
        for i, target in enumerate(foot_trajectory):
            # Use previous solution as initialization for smoothness
            q_guess = q_trajectory[i-1] if i > 0 else (q_init[i] if q_init is not None else None)
            
            q_sol, success, error = self.inverse_kinematics(
                target, side=side, q_init=q_guess, max_iter=50
            )
            
            q_trajectory[i] = q_sol
            success_mask[i] = success
            error_array[i] = error
            
            if (i + 1) % 50 == 0:
                success_rate = np.sum(success_mask[:i+1]) / (i + 1) * 100
                print(f"  Point {i+1}/{n_points}: Success rate: {success_rate:.1f}%, "
                      f"Max error: {error_array[:i+1].max()*1000:.2f}mm")
        
        success_rate = np.sum(success_mask) / n_points * 100
        print(f"\nIK Solution Summary ({side}):")
        print(f"  Success rate: {success_rate:.1f}% ({np.sum(success_mask)}/{n_points})")
        print(f"  Max error: {error_array.max()*1000:.2f} mm")
        print(f"  Mean error: {error_array.mean()*1000:.2f} mm")
        
        return q_trajectory, success_mask, error_array


def main():
    """Generate IK solutions for walking trajectories."""
    
    print("="*80)
    print("BIPED ROBOT IK SOLVER")
    print("="*80)
    
    # Load trajectories
    print("\nLoading trajectories...")
    base_traj = np.load("base_trajectory.npy")
    foot1_traj = np.load("foot1_trajectory.npy")
    foot2_traj = np.load("foot2_trajectory.npy")
    
    print(f"  Base trajectory: {base_traj.shape}")
    print(f"  Foot1 trajectory: {foot1_traj.shape}")
    print(f"  Foot2 trajectory: {foot2_traj.shape}")
    
    # Initialize IK solver
    print("\nInitializing IK solver...")
    model_path = "legged_robot_ik.xml"
    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}")
        return
    
    solver = BipedIKSolver(model_path)
    print(f"✓ Model loaded successfully")
    print(f"  Left leg joints: {len(solver.leg_joints.get('left', []))} DOF")
    print(f"  Right leg joints: {len(solver.leg_joints.get('right', []))} DOF")
    
    # Solve IK for both feet
    print("\n" + "="*80)
    print("SOLVING INVERSE KINEMATICS")
    print("="*80)
    
    q_left_traj, left_success, left_errors = solver.solve_trajectory(
        foot1_traj, side='left'
    )
    
    q_right_traj, right_success, right_errors = solver.solve_trajectory(
        foot2_traj, side='right'
    )
    
    # Save solutions
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    np.save("left_leg_angles.npy", q_left_traj)
    np.save("right_leg_angles.npy", q_right_traj)
    np.save("left_leg_success.npy", left_success)
    np.save("right_leg_success.npy", right_success)
    np.save("left_leg_errors.npy", left_errors)
    np.save("right_leg_errors.npy", right_errors)
    
    print(f"✓ left_leg_angles.npy")
    print(f"✓ right_leg_angles.npy")
    print(f"✓ left_leg_success.npy")
    print(f"✓ right_leg_success.npy")
    print(f"✓ left_leg_errors.npy")
    print(f"✓ right_leg_errors.npy")
    
    print(f"\n{'='*80}")
    print("IK SOLVING COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
