#!/usr/bin/env python3
"""
MuJoCo Simulation for Biped Walking

Simulates biped robot walking with PD controllers on joint trajectories.
"""

import numpy as np
import mujoco
from mujoco import viewer
from pathlib import Path

class BipedWalkerController:
    """PD controller for biped walking."""
    
    def __init__(self, model_path, joint_config=None):
        """
        Initialize controller.
        
        Args:
            model_path: Path to URDF/XML model
            joint_config: Dict mapping joint names to indices
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Default PD gains
        self.kp = 100.0  # Proportional gain
        self.kd = 10.0   # Derivative gain
        
        # Find controlled joints
        self.controlled_joints = []
        for i, name in enumerate(self.model.joint_names):
            if 'leg' in name.lower() or 'hip' in name.lower() or 'knee' in name.lower() or 'ankle' in name.lower():
                self.controlled_joints.append(i)
        
        print(f"Found {len(self.controlled_joints)} controlled joints")
        
        # Trajectory data
        self.q_trajectory = None
        self.timestep = 0
    
    def load_trajectories(self):
        """Load pre-computed IK trajectories."""
        try:
            self.q_left = np.load("left_leg_angles.npy")
            self.q_right = np.load("right_leg_angles.npy")
            self.total_steps = len(self.q_left)
            print(f"✓ Loaded trajectories: {self.total_steps} steps")
            return True
        except FileNotFoundError as e:
            print(f"✗ Error loading trajectories: {e}")
            return False
    
    def get_desired_angles(self, step):
        """Get desired joint angles for current step."""
        if self.q_left is None:
            return None, None
        
        # Wrap around if needed
        step = step % self.total_steps
        
        return self.q_left[step], self.q_right[step]
    
    def control_step(self, step):
        """
        Compute control torques for current step.
        
        Args:
            step: Current simulation step
        """
        q_left_des, q_right_des = self.get_desired_angles(step)
        
        if q_left_des is None:
            return
        
        # Get current joint angles and velocities
        q_current = self.data.qpos[7:]  # Skip base (7 DOF: 3 pos + 4 quat)
        qvel_current = self.data.qvel[6:]  # Skip base velocity (6 DOF)
        
        # Build desired angles (assuming specific joint ordering)
        # Adjust based on your actual model structure
        if len(q_current) >= 6:
            q_desired = np.concatenate([q_left_des, q_right_des])[:len(q_current)]
            
            # PD control: tau = kp * (q_des - q) - kd * qvel
            error = q_desired - q_current[:len(q_desired)]
            vel_error = -qvel_current[:len(q_desired)]
            
            tau = self.kp * error + self.kd * vel_error
            
            # Apply torques
            self.data.ctrl[:len(tau)] = np.clip(tau, -100, 100)  # Torque limits
    
    def simulate(self, duration=10.0, render=False):
        """
        Run simulation.
        
        Args:
            duration: Simulation time in seconds
            render: Whether to display visualization
        """
        if not self.load_trajectories():
            return False
        
        dt = self.model.opt.timestep
        n_steps = int(duration / dt)
        
        print(f"\nSimulating for {duration}s ({n_steps} steps)...")
        print(f"  Timestep: {dt*1000:.2f} ms")
        
        # Reset to initial state
        self.data.qpos[2] = 0.210  # Initial hip height
        mujoco.mj_forward(self.model, self.data)
        
        # Open viewer if requested
        if render:
            with viewer.launch_passive(self.model, self.data) as v:
                while v.is_running() and self.timestep < n_steps:
                    self.control_step(self.timestep)
                    mujoco.mj_step(self.model, self.data)
                    self.timestep += 1
                    v.sync()
        else:
            for step in range(n_steps):
                self.control_step(step)
                mujoco.mj_step(self.model, self.data)
                
                if (step + 1) % 1000 == 0:
                    print(f"  Step {step+1}/{n_steps}")
        
        print(f"✓ Simulation complete")
        return True


def main():
    """Run walking simulation."""
    
    print("="*80)
    print("BIPED ROBOT WALKING SIMULATION")
    print("="*80)
    
    model_path = "legged_robot_ik.xml"
    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}")
        return
    
    # Initialize controller
    controller = BipedWalkerController(model_path)
    
    # Run simulation
    print("\nStarting simulation...")
    try:
        controller.simulate(duration=5.0, render=False)
        print("\n✓ Walking simulation successful!")
    except Exception as e:
        print(f"\n✗ Simulation error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
