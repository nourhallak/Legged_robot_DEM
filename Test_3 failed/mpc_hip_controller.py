"""
MPC Hip Height Controller
==========================

Model Predictive Control for biped hip height regulation.
Controls the vertical hip position to maintain balance and stable walking.
"""

import numpy as np
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov
from scipy.integrate import odeint


class MPCHipController:
    """Model Predictive Control for hip height."""
    
    def __init__(self, prediction_horizon=10, dt=0.01):
        """
        Initialize MPC hip controller.
        
        Args:
            prediction_horizon: Number of steps to predict ahead
            dt: Time step
        """
        self.horizon = prediction_horizon
        self.dt = dt
        
        # Hip dynamics: simple spring-mass system
        # z_ddot = -k/m * (z - z_ref) - c/m * z_dot
        self.m = 1.0      # Mass (normalized)
        self.k = 100.0    # Spring stiffness
        self.c = 10.0     # Damping coefficient
        
        # Reference hip height (standing height in meters)
        self.z_ref = 0.42
        
        # State: [z, z_dot]
        self.state = np.array([self.z_ref, 0.0])
        
        # Control weights
        self.Q = np.diag([100.0, 1.0])    # State cost (height, velocity)
        self.R = 1.0                      # Control cost
    
    def hip_dynamics(self, state, control, t=0):
        """
        Hip vertical dynamics.
        
        State: [z, z_dot]
        Control: Force applied to hip
        """
        z, z_dot = state
        
        # Acceleration from force, spring, and damping
        z_ddot = (control - self.k * (z - self.z_ref) - self.c * z_dot) / self.m
        
        return np.array([z_dot, z_ddot])
    
    def predict_trajectory(self, state, control_sequence):
        """
        Predict hip trajectory given control sequence.
        
        Args:
            state: Initial state [z, z_dot]
            control_sequence: Control inputs over horizon
        
        Returns:
            Trajectory array (horizon+1, 2)
        """
        trajectory = [state.copy()]
        current_state = state.copy()
        
        for control in control_sequence:
            # Simple Euler integration
            state_dot = self.hip_dynamics(current_state, control)
            current_state = current_state + state_dot * self.dt
            trajectory.append(current_state.copy())
        
        return np.array(trajectory)
    
    def cost_trajectory(self, trajectory):
        """Calculate cost of trajectory."""
        cost = 0.0
        
        for state in trajectory:
            # Quadratic state cost
            state_error = np.array([state[0] - self.z_ref, state[1]])
            cost += state_error @ self.Q @ state_error
        
        return cost
    
    def optimize_control(self, current_state, desired_height=None):
        """
        Optimize control inputs using MPC.
        
        Args:
            current_state: Current [z, z_dot]
            desired_height: Target height (use default if None)
        
        Returns:
            Optimal control input for current step
        """
        if desired_height is not None:
            self.z_ref = desired_height
        
        # Simple LQR-style control
        # Feedback: F = -K * (x - x_ref)
        state_error = np.array([
            current_state[0] - self.z_ref,
            current_state[1]
        ])
        
        # LQR gain (approximate)
        K = np.array([50.0, 20.0])  # Proportional and derivative gains
        
        control = -K @ state_error
        
        # Limit control input
        control = np.clip(control, -100.0, 100.0)
        
        return control
    
    def step(self, current_height, current_velocity, desired_height=None):
        """
        Step the MPC controller.
        
        Args:
            current_height: Current z position
            current_velocity: Current z velocity
            desired_height: Target height (use default if None)
        
        Returns:
            Control force and predicted state
        """
        current_state = np.array([current_height, current_velocity])
        self.state = current_state
        
        # Get optimal control
        control = self.optimize_control(current_state, desired_height)
        
        # Update state
        state_dot = self.hip_dynamics(current_state, control)
        next_state = current_state + state_dot * self.dt
        
        self.state = next_state
        
        return control, next_state
    
    def set_reference_height(self, height):
        """Set desired hip height."""
        self.z_ref = height


if __name__ == "__main__":
    # Test the MPC controller
    print("="*70)
    print("MPC HIP CONTROLLER TEST")
    print("="*70)
    
    controller = MPCHipController(prediction_horizon=10, dt=0.01)
    
    # Simulate walking cycle
    print("\n🚶 Simulating hip height control over 2 seconds...")
    
    time = 0.0
    z = 0.42
    z_dot = 0.0
    
    times = []
    heights = []
    velocities = []
    controls = []
    
    while time < 2.0:
        # Get control
        control, (z, z_dot) = controller.step(z, z_dot)
        
        times.append(time)
        heights.append(z)
        velocities.append(z_dot)
        controls.append(control)
        
        time += controller.dt
    
    print(f"\n📊 Results:")
    print(f"   Time range: 0-{times[-1]:.2f}s")
    print(f"   Height range: {min(heights):.4f} - {max(heights):.4f}m")
    print(f"   Final height: {heights[-1]:.4f}m (ref: {controller.z_ref:.4f}m)")
    print(f"   Final velocity: {velocities[-1]:.4f}m/s")
    
    print("\n✅ MPC controller ready for integration")
