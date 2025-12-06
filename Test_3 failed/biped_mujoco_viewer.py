"""
MuJoCo Biped Walker - Visualize 10-Step Sinusoidal Gait with MPC Hip Control
==============================================================================

Real-time visualization of 10-step sinusoidal walking with IK-solved joint angles
and MPC-controlled hip height in MuJoCo physics engine.
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path
from mpc_hip_controller import MPCHipController


class BipedMuJoCoWalker:
    """Simulate biped robot walking in MuJoCo."""
    
    def __init__(self):
        """Initialize MuJoCo model and load trajectories."""
        print("="*70)
        print("BIPED MUJOCU WALKER - 10-STEP SINUSOIDAL GAIT")
        print("="*70)
        
        # Load MuJoCo model
        print("\n📂 Loading MuJoCo model...")
        model_path = Path(__file__).parent / "legged_robot_ik.xml"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        
        print(f"✅ Model loaded: {model_path.name}")
        print(f"   DOF: {self.model.nq}")
        print(f"   Actuators: {self.model.nu}")
        
        # Load IK solutions and trajectories
        print("\n📂 Loading trajectories and IK solutions...")
        self.load_trajectories()
        
        # Initialize MPC hip controller
        print("\n🎮 Initializing MPC hip height controller...")
        self.mpc_controller = MPCHipController(prediction_horizon=10, dt=0.01)
        self.desired_hip_height = 0.42  # Standing height
        
        # Control parameters
        self.current_step = 0
        self.paused = False
        self.slow_motion_factor = 1.0
        self.simulation_time = 0.0
        self.total_steps = 0
    
    def load_trajectories(self):
        """Load trajectories and IK solutions."""
        # Load trajectories
        traj_file = Path(__file__).parent / "biped_10step_sinusoidal.npy"
        traj_data = np.load(str(traj_file), allow_pickle=True).item()
        self.traj_times = traj_data['times']
        self.total_duration = traj_data['total_duration']
        self.left_trajectory = traj_data['left_trajectory']
        self.right_trajectory = traj_data['right_trajectory']
        
        # Load IK solutions
        ik_file = Path(__file__).parent / "biped_ik_solutions.npy"
        ik_data = np.load(str(ik_file), allow_pickle=True).item()
        self.left_angles = ik_data['left_joint_angles']
        self.right_angles = ik_data['right_joint_angles']
        
        self.total_steps = len(self.traj_times)
        
        print(f"✅ Trajectories loaded: {self.total_steps} points")
        print(f"   Duration: {self.total_duration:.2f}s")
        print(f"   Left joint angles shape: {self.left_angles.shape}")
        print(f"   Right joint angles shape: {self.right_angles.shape}")
    
    def get_joint_targets(self, time_idx):
        """Get target joint angles for current time step."""
        if time_idx >= len(self.left_angles):
            time_idx = len(self.left_angles) - 1
        
        # Get angles for both legs
        left_hip, left_knee, left_ankle = self.left_angles[time_idx]
        right_hip, right_knee, right_ankle = self.right_angles[time_idx]
        
        return {
            'left_hip': left_hip,
            'left_knee': left_knee,
            'left_ankle': left_ankle,
            'right_hip': right_hip,
            'right_knee': right_knee,
            'right_ankle': right_ankle
        }
    
    def apply_joint_control(self, targets, time_idx=0):
        """Apply direct position control to track joint targets."""
        # Use direct position setting for accurate tracking
        self.data.qpos[3] = targets['left_hip']       # Left hip
        self.data.qpos[4] = targets['left_knee']      # Left knee
        self.data.qpos[5] = targets['left_ankle']     # Left ankle
        self.data.qpos[6] = targets['right_hip']      # Right hip
        self.data.qpos[7] = targets['right_knee']     # Right knee
        self.data.qpos[8] = targets['right_ankle']    # Right ankle
        
        # Hip X and Y controlled by trajectory
        # Hip Z (height) controlled by MPC
        current_idx = time_idx % len(self.traj_times)
        
        # Get current foot X positions from trajectory
        left_foot_x = self.left_trajectory[current_idx, 0]
        right_foot_x = self.right_trajectory[current_idx, 0]
        
        # Hip X position: weighted average that favors the forward-moving leg
        if left_foot_x > right_foot_x:
            # Left leg swinging forward - hip follows left leg more (70% left, 30% right)
            hip_x = 0.7 * left_foot_x + 0.3 * right_foot_x
        else:
            # Right leg swinging forward - hip follows right leg more (70% right, 30% left)
            hip_x = 0.7 * right_foot_x + 0.3 * left_foot_x
        
        # Get current hip height and velocity
        current_hip_z = self.data.qpos[2]
        current_hip_z_vel = self.data.qvel[2]
        
        # MPC controller determines Z position
        _, (mpc_z, mpc_z_vel) = self.mpc_controller.step(
            current_hip_z, 
            current_hip_z_vel, 
            self.desired_hip_height
        )
        
        # Set hip position: X and Y from trajectory, Z from MPC
        self.data.qpos[0] = hip_x           # Hip X - controlled by trajectory
        self.data.qpos[1] = -0.00497        # Hip Y - fixed lateral position
        self.data.qpos[2] = mpc_z           # Hip Z - controlled by MPC
    
    def step_simulation(self, time_idx):
        """Step simulation with interpolation between time points."""
        if time_idx >= len(self.traj_times):
            return False  # Simulation ended
        
        # Get targets for current step
        targets = self.get_joint_targets(time_idx)
        
        # Apply control
        self.apply_joint_control(targets, time_idx)
        
        # Step physics
        mujoco.mj_step(self.model, self.data)
        
        return True
    
    def run_viewer(self):
        """Run interactive MuJoCo viewer with keyboard controls."""
        print("\n" + "="*70)
        print("MUJOCU VIEWER - BIPED WALKING")
        print("="*70)
        print("\n🎮 Controls in Viewer:")
        print("   Mouse: Click and drag to rotate view")
        print("   Scroll: Zoom in/out")
        print("   X/Y/Z: View from each axis")
        
        # Initialize to first pose
        print(f"\n🔧 Initializing robot to walking pose...")
        for init_step in range(200):
            targets = self.get_joint_targets(0)  # Use first frame
            self.apply_joint_control(targets, 0)
            mujoco.mj_step(self.model, self.data)
        
        print(f"✅ Robot initialized")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Configure viewer
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -25
            
            print(f"\n📺 Viewer launched - watch for motion!")
            print(f"   (Looping continuously - close window to exit)")
            
            time_step_idx = 0
            total_steps_simulated = 0
            last_print = 0
            gait_cycle = 0
            
            while viewer.is_running():
                with viewer.lock():
                    # Get and apply targets
                    targets = self.get_joint_targets(time_step_idx)
                    self.apply_joint_control(targets, time_step_idx)
                    
                    # Step simulation
                    mujoco.mj_step(self.model, self.data)
                    time_step_idx += 1
                    total_steps_simulated += 1
                    
                    # Loop back to start when cycle completes
                    if time_step_idx >= len(self.traj_times):
                        time_step_idx = 0
                        gait_cycle += 1
                        print(f"   ↻ Gait cycle {gait_cycle} complete - restarting")
                    
                    # Print status every 200 steps
                    if total_steps_simulated - last_print >= 200:
                        current_time = self.traj_times[time_step_idx - 1] if time_step_idx > 0 else 0
                        print(f"   Cycle {gait_cycle} | Step {time_step_idx:4d}/{len(self.traj_times)}")
                        last_print = total_steps_simulated
                    
                    viewer.sync()
                
                # Sleep briefly
                time.sleep(0.001)
    
    def run_headless(self, num_episodes=1):
        """Run simulation without viewer (for testing/data collection)."""
        print(f"\n▶️  Running {num_episodes} episode(s) in headless mode...")
        
        for episode in range(num_episodes):
            print(f"\n📍 Episode {episode + 1}/{num_episodes}")
            
            # Reset
            mujoco.mj_resetData(self.model, self.data)
            time_step_idx = 0
            
            # Run simulation
            while time_step_idx < len(self.traj_times):
                targets = self.get_joint_targets(time_step_idx)
                self.apply_joint_control(targets, time_step_idx)
                mujoco.mj_step(self.model, self.data)
                time_step_idx += 1
                
                if time_step_idx % 200 == 0:
                    current_time = self.traj_times[time_step_idx - 1]
                    progress = (time_step_idx / len(self.traj_times)) * 100
                    print(f"   Progress: {progress:.1f}% | Time: {current_time:.2f}s")
            
            print(f"✅ Episode {episode + 1} complete")


def main():
    """Main entry point."""
    try:
        # Create walker
        walker = BipedMuJoCoWalker()
        
        # Run viewer
        print("\n" + "="*70)
        print("LAUNCHING MUJOCU VIEWER")
        print("="*70)
        walker.run_viewer()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  - legged_robot_ik.xml (MuJoCo model)")
        print("  - biped_10step_sinusoidal.npy (trajectories)")
        print("  - biped_ik_solutions.npy (IK solutions)")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
