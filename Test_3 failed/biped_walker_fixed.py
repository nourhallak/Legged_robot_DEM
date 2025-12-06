"""
Fixed MuJoCo Biped Walker - Proper Standing + Walking
======================================================

Starts from proper standing position, then executes 10-step walking.
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path


class BipedMuJoCoWalkerFixed:
    """Simulate biped robot walking in MuJoCo - FIXED VERSION."""
    
    def __init__(self):
        """Initialize MuJoCo model and load trajectories."""
        print("="*70)
        print("BIPED MUJOCU WALKER - FIXED (10-STEP SINUSOIDAL GAIT)")
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
        
        # Robot parameters
        self.link_2_length = 0.014  # Thigh
        self.link_1_length = 0.014  # Shin
        self.hip_height = 0.42
        self.standing_height = 0.392
        
        # Control parameters
        self.current_step = 0
    
    def load_trajectories(self):
        """Load trajectories and IK solutions."""
        # Load trajectories
        traj_file = Path(__file__).parent / "biped_10step_sinusoidal.npy"
        traj_data = np.load(str(traj_file), allow_pickle=True).item()
        self.traj_times = traj_data['times']
        self.total_duration = traj_data['total_duration']
        self.left_foot_traj = traj_data['left_trajectory']
        self.right_foot_traj = traj_data['right_trajectory']
        
        # Load IK solutions
        ik_file = Path(__file__).parent / "biped_ik_solutions.npy"
        ik_data = np.load(str(ik_file), allow_pickle=True).item()
        self.left_angles = ik_data['left_joint_angles']
        self.right_angles = ik_data['right_joint_angles']
        
        self.total_steps = len(self.traj_times)
        
        print(f"✅ Trajectories loaded: {self.total_steps} points")
        print(f"   Duration: {self.total_duration:.2f}s")
    
    def get_standing_pose(self):
        """Get joint angles for neutral standing position."""
        # Standing pose: legs fully extended downward
        # For a 3-link leg: hip=0, knee=0, ankle=0 means leg points down
        return {
            'left_hip': 0.0,
            'left_knee': 0.0,
            'left_ankle': 0.0,
            'right_hip': 0.0,
            'right_knee': 0.0,
            'right_ankle': 0.0,
        }
    
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
            'right_ankle': right_ankle,
        }
    
    def apply_joint_control(self, targets, kp=100.0, kd=10.0):
        """Apply PD control to track joint targets."""
        motor_names = [
            'hip_link_2_1_motor',
            'link_2_1_link_1_1_motor',
            'link_1_1_foot_1_motor',
            'hip_link_2_2_motor',
            'link_2_2_link_1_2_motor',
            'link_1_2_foot_2_motor',
        ]
        
        target_list = [
            targets['left_hip'],
            targets['left_knee'],
            targets['left_ankle'],
            targets['right_hip'],
            targets['right_knee'],
            targets['right_ankle'],
        ]
        
        for motor_idx, (motor_name, target_angle) in enumerate(zip(motor_names, target_list)):
            motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_name)
            
            if motor_id >= 0:
                joint_id = self.model.actuator_trnid[motor_id, 0]
                
                if joint_id >= 0 and joint_id < len(self.data.qpos):
                    current_angle = self.data.qpos[joint_id]
                    current_vel = self.data.qvel[joint_id] if joint_id < len(self.data.qvel) else 0.0
                    
                    error = target_angle - current_angle
                    control = kp * error - kd * current_vel
                    control = np.clip(control, -1.0, 1.0)
                    
                    self.data.ctrl[motor_id] = control
    
    def settle_to_standing(self, num_steps=500):
        """Let robot settle into standing position."""
        print(f"📍 Settling to standing position ({num_steps} steps)...")
        standing_pose = self.get_standing_pose()
        
        for step in range(num_steps):
            self.apply_joint_control(standing_pose, kp=100.0, kd=15.0)
            mujoco.mj_step(self.model, self.data)
            
            if step % 100 == 0:
                left_foot_z = self.left_foot_traj[0, 2]  # Expected height
                print(f"   Step {step:3d}: Settling... Left foot Z: {self.data.xpos[5][2]:.4f}m (target: {left_foot_z:.4f}m)")
        
        print(f"✅ Robot settled to standing")
    
    def run_viewer(self):
        """Run interactive MuJoCo viewer with walking."""
        print("\n" + "="*70)
        print("MUJOCU VIEWER - BIPED WALKING")
        print("="*70)
        print("\n🎮 Controls in Viewer:")
        print("   Mouse: Click and drag to rotate view")
        print("   Scroll: Zoom in/out")
        print("   T: Toggle transparency")
        print("   F: Toggle frame rate")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Configure viewer
            viewer.cam.distance = 1.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            
            # Step 1: Settle to standing
            self.settle_to_standing(num_steps=500)
            
            print(f"\n🚶 Starting walking trajectory...")
            print(f"   Robot will take 10 steps over {self.total_duration:.2f} seconds")
            
            time_step_idx = 0
            total_steps_simulated = 500  # Account for settling steps
            last_print = 0
            
            while viewer.is_running() and time_step_idx < len(self.traj_times):
                with viewer.lock():
                    # Get and apply targets
                    targets = self.get_joint_targets(time_step_idx)
                    self.apply_joint_control(targets, kp=100.0, kd=10.0)
                    
                    # Step simulation
                    mujoco.mj_step(self.model, self.data)
                    time_step_idx += 1
                    total_steps_simulated += 1
                    
                    # Print status every 200 steps
                    if total_steps_simulated - last_print >= 200:
                        current_time = self.traj_times[time_step_idx - 1]
                        progress = (time_step_idx / len(self.traj_times)) * 100
                        
                        # Get foot positions
                        left_foot_z = self.data.xpos[5][2] if len(self.data.xpos) > 5 else 0
                        right_foot_z = self.data.xpos[8][2] if len(self.data.xpos) > 8 else 0
                        
                        print(f"   Step {time_step_idx:4d}/{len(self.traj_times)} | Time: {current_time:6.2f}s | Progress: {progress:5.1f}% | L-Z: {left_foot_z:.4f}m | R-Z: {right_foot_z:.4f}m")
                        last_print = total_steps_simulated
                    
                    viewer.sync()
                
                # Sleep briefly
                time.sleep(0.001)
            
            print(f"\n✅ Walking complete!")
            print(f"   Total steps simulated: {total_steps_simulated}")
            print(f"   (Viewer will close in 3 seconds)")
            
            # Keep viewer open for a bit
            for _ in range(30):
                with viewer.lock():
                    viewer.sync()
                time.sleep(0.1)


def main():
    """Main entry point."""
    try:
        # Create walker
        walker = BipedMuJoCoWalkerFixed()
        
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
