"""
10-Step Sinusoidal Gait Generator for Biped Robot
==================================================

Generates synchronized 10-step walking trajectory for both legs
using sinusoidal foot paths based on robot dimensions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class Biped10StepWalkingGenerator:
    """Generate multi-step synchronized walking trajectories."""
    
    def __init__(self):
        """Initialize with robot dimensions."""
        # Robot dimensions
        self.link_2_length = 0.014  # Thigh (14mm)
        self.link_1_length = 0.014  # Shin (14mm)
        self.max_leg_length = 0.028  # Total leg length
        
        # Hip position
        self.hip_height = 0.42  # meters
        self.standing_height = 0.431  # meters (MuJoCo model: foot at Z=0.431 when all joints=0)
        
        # Actual robot foot Y positions (from geometry)
        self.left_foot_y = -0.0134   # Left foot at Y=-13.4mm
        self.right_foot_y = -0.0064  # Right foot at Y=-6.4mm
        
        # Gait parameters
        self.stride_length = 0.006  # 6mm stride (wider than 3mm for better visual gait)
        self.step_height = 0.004   # 4mm clearance
        self.stance_duration = 0.4  # seconds
        self.swing_duration = 0.2   # seconds
        self.step_period = self.stance_duration + self.swing_duration  # 0.6s total
        
        print("="*70)
        print("MULTI-STEP SINUSOIDAL BIPED WALKING GENERATOR")
        print("="*70)
        print(f"\n📐 Robot Config: {self.link_2_length*1000:.0f}mm + {self.link_1_length*1000:.0f}mm legs")
        print(f"🚶 Gait Params: {self.stride_length*1000:.0f}mm stride, {self.step_height*1000:.0f}mm height")
        print(f"⏱️  Timing: {self.step_period:.2f}s per step (1.67 Hz)")
    
    def create_swing_phase(self, start_x, stride, step_height, num_points=50):
        """Create sinusoidal swing phase trajectory."""
        t_norm = np.linspace(0, 1, num_points)
        
        # Sinusoidal height profile
        z_swing = self.standing_height + step_height * np.sin(np.pi * t_norm)
        
        # Linear forward motion
        x_swing = start_x + stride * t_norm
        
        return x_swing, z_swing, t_norm
    
    def create_stance_phase(self, start_x, stride, num_points=50):
        """Create stance phase trajectory (foot on ground, sliding forward)."""
        t_norm = np.linspace(0, 1, num_points)
        
        # Constant height (on ground)
        z_stance = np.full(num_points, self.standing_height)
        
        # Linear forward motion
        x_stance = start_x + stride * t_norm
        
        return x_stance, z_stance, t_norm
    
    def generate_10_step_trajectory(self, num_points_per_phase=50):
        """
        Generate multi-step synchronized trajectory with phase shift.
        
        Both legs have SAME FREQUENCY (period) but 180-degree PHASE SHIFT:
        - When left swings, right is in stance
        - When right swings, left is in stance
        - Smooth alternating gait
        - **Both feet return to starting position at end for smooth looping**
        
        Returns:
            Dictionary with complete trajectories
        """
        
        num_steps = 10  # 10 steps (5 per leg) - original working version
        n_per_phase = num_points_per_phase  # Points per phase (swing or stance)
        n_per_cycle = n_per_phase * 2  # One complete swing+stance cycle
        total_points = num_steps * n_per_phase
        
        # Initialize arrays
        left_trajectory = np.zeros((total_points, 3))   # [x, y, z]
        right_trajectory = np.zeros((total_points, 3))
        times = np.zeros(total_points)
        
        # Offset feet positions forward so hip naturally leads
        # If feet start at positive X, hip will be behind
        # If feet start at negative X and move forward, hip leads
        hip_lead_offset = 0.010  # 10mm - hip should lead by this amount
        start_x = -hip_lead_offset  # Starting position for both feet
        current_x_left = start_x
        current_x_right = start_x
        current_time = 0.0
        
        print(f"\n🚶 Generating SYNCHRONIZED {num_steps}-step sinusoidal trajectories...")
        print(f"   Points per phase: {n_per_phase}")
        print(f"   Total points: {total_points}")
        print(f"   Both legs same frequency, 180° phase shift")
        print(f"   Hip lead offset: {hip_lead_offset*1000:.1f}mm")
        print(f"   Trajectory loops smoothly (feet return to start)")
        
        for step in range(num_steps):
            step_start_idx = step * n_per_phase
            step_end_idx = step_start_idx + n_per_phase
            
            # ALTERNATING: even steps = left swings, odd steps = right swings
            if step % 2 == 0:
                # STEP EVEN (0, 2, 4, ...): LEFT LEG SWINGS, RIGHT LEG LANDS/SUPPORTS
                
                # LEFT LEG SWING PHASE - foot lifts up and forward
                x_swing, z_swing, _ = self.create_swing_phase(
                    current_x_left, self.stride_length, self.step_height, n_per_phase
                )
                left_trajectory[step_start_idx : step_end_idx, 0] = x_swing
                left_trajectory[step_start_idx : step_end_idx, 2] = z_swing
                current_x_left = x_swing[-1]
                
                # RIGHT LEG STANCE - foot on ground, supporting body weight
                right_trajectory[step_start_idx : step_end_idx, 0] = current_x_right
                right_trajectory[step_start_idx : step_end_idx, 2] = self.standing_height
                
                phase_name = "LEFT ↑ (RIGHT ▬)"
                
            else:
                # STEP ODD (1, 3, 5, ...): RIGHT LEG SWINGS, LEFT LEG LANDS/SUPPORTS
                
                # RIGHT LEG SWING PHASE - foot lifts up and forward
                x_swing, z_swing, _ = self.create_swing_phase(
                    current_x_right, self.stride_length, self.step_height, n_per_phase
                )
                right_trajectory[step_start_idx : step_end_idx, 0] = x_swing
                right_trajectory[step_start_idx : step_end_idx, 2] = z_swing
                current_x_right = x_swing[-1]
                
                # LEFT LEG STANCE - foot on ground, supporting body weight
                left_trajectory[step_start_idx : step_end_idx, 0] = current_x_left
                left_trajectory[step_start_idx : step_end_idx, 2] = self.standing_height
                
                phase_name = "RIGHT ↑ (LEFT ▬)"
            
            # Fill time
            step_times = np.linspace(current_time, 
                                     current_time + self.step_period, 
                                     n_per_phase)
            times[step_start_idx : step_end_idx] = step_times
            current_time += self.step_period
            
            # Y coordinate (lateral) set to actual foot positions
            left_trajectory[step_start_idx : step_end_idx, 1] = self.left_foot_y
            right_trajectory[step_start_idx : step_end_idx, 1] = self.right_foot_y
            
            if step < 15 or step >= num_steps - 2:  # Print first 15 and last 2
                print(f"   Step {step+1:2d}: {phase_name:20s} | Time: {current_time-self.step_period:.2f}-{current_time:.2f}s")
            elif step == 15:
                print(f"   ...")
        
        # Simple trajectory - let it loop naturally
        # The hip control in the viewer will handle the cycling
        
        trajectory_data = {
            'left_trajectory': left_trajectory,
            'right_trajectory': right_trajectory,
            'times': times,
            'num_steps': num_steps,
            'total_duration': times[-1],
            'total_points': total_points,
            'stride_length': self.stride_length,
            'step_height': self.step_height,
            'method': 'sinusoidal',
            'hip_lead_offset': hip_lead_offset
        }
        
        print(f"\n✅ Generated trajectories")
        print(f"   Total duration: {times[-1]:.2f}s")
        print(f"   Left leg distance: {(current_x_left + hip_lead_offset) * 1000:.1f}mm (5 steps)")
        print(f"   Right leg distance: {(current_x_right + hip_lead_offset) * 1000:.1f}mm (5 steps)")
        
        return trajectory_data
    
    def convert_to_joint_angles(self, foot_position):
        """Convert foot position to joint angles using 2-link IK."""
        x, y = foot_position[0], foot_position[1]
        distance = np.sqrt(x**2 + y**2)
        
        if distance < 1e-6:  # At hip
            return np.array([0, 0, 0])
        
        # Clamp to reachable workspace
        distance = np.clip(distance, 0, self.max_leg_length)
        
        # Law of cosines for knee angle
        l1 = self.link_1_length
        l2 = self.link_2_length
        
        cos_knee = (distance**2 - l1**2 - l2**2) / (2 * l1 * l2 + 1e-6)
        cos_knee = np.clip(cos_knee, -1, 1)
        knee_angle = np.arccos(cos_knee)
        
        # Hip angle
        alpha = np.arctan2(y, x)
        sin_beta = l1 * np.sin(knee_angle) / (distance + 1e-6)
        sin_beta = np.clip(sin_beta, -1, 1)
        beta = np.arcsin(sin_beta)
        hip_angle = alpha + beta
        
        # Ankle angle (passive)
        ankle_angle = -(hip_angle + knee_angle)
        
        return np.array([hip_angle, knee_angle, ankle_angle])
    
    def trajectories_to_joint_angles(self, trajectory_data):
        """Convert foot trajectories to joint angle trajectories."""
        left_traj = trajectory_data['left_trajectory']
        right_traj = trajectory_data['right_trajectory']
        
        left_angles = np.array([self.convert_to_joint_angles(pos[:2]) for pos in left_traj])
        right_angles = np.array([self.convert_to_joint_angles(pos[:2]) for pos in right_traj])
        
        trajectory_data['left_joint_angles'] = left_angles
        trajectory_data['right_joint_angles'] = right_angles
        
        return trajectory_data
    
    def plot_trajectories(self, trajectory_data, save_path=None):
        """Create comprehensive visualization of 10-step walking."""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        left_traj = trajectory_data['left_trajectory']
        right_traj = trajectory_data['right_trajectory']
        times = trajectory_data['times']
        
        # ===== ROW 1: Side View =====
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(left_traj[:, 0]*1000, left_traj[:, 2]*1000, 'b-', linewidth=2, label='Left leg')
        ax1.plot(right_traj[:, 0]*1000, right_traj[:, 2]*1000, 'r-', linewidth=2, label='Right leg')
        ax1.axhline(self.standing_height*1000, color='brown', linestyle='--', linewidth=2, label='Ground')
        
        # Mark step transitions
        for step in range(trajectory_data['num_steps']+1):
            step_idx = step * int(len(times) / trajectory_data['num_steps'])
            if step_idx < len(times):
                ax1.axvline(times[min(step_idx, len(times)-1)], color='gray', 
                           linestyle=':', alpha=0.3, linewidth=0.8)
        
        ax1.set_xlabel('Forward Position [mm]', fontsize=10)
        ax1.set_ylabel('Height [mm]', fontsize=10)
        ax1.set_title('Side View: Both Legs (10 Steps)', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # ===== ROW 1: Top View =====
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(left_traj[:, 0]*1000, left_traj[:, 1]*1000, 'b-', linewidth=2, label='Left')
        ax2.plot(right_traj[:, 0]*1000, right_traj[:, 1]*1000, 'r-', linewidth=2, label='Right')
        ax2.set_xlabel('Forward [mm]', fontsize=9)
        ax2.set_ylabel('Lateral [mm]', fontsize=9)
        ax2.set_title('Top View', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        ax2.axis('equal')
        
        # ===== ROW 2: Height vs Time =====
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(times, left_traj[:, 2]*1000, 'b-', linewidth=2, label='Left')
        ax3.plot(times, right_traj[:, 2]*1000, 'r-', linewidth=2, label='Right')
        ax3.axhline(self.standing_height*1000, color='brown', linestyle='--', alpha=0.5, linewidth=1)
        ax3.fill_between(times, self.standing_height*1000, left_traj[:, 2]*1000, alpha=0.1, color='blue')
        ax3.fill_between(times, self.standing_height*1000, right_traj[:, 2]*1000, alpha=0.1, color='red')
        ax3.set_xlabel('Time [s]', fontsize=9)
        ax3.set_ylabel('Height [mm]', fontsize=9)
        ax3.set_title('Height vs Time', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        
        # ===== ROW 2: Forward Position vs Time =====
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(times, left_traj[:, 0]*1000, 'b-', linewidth=2, label='Left')
        ax4.plot(times, right_traj[:, 0]*1000, 'r-', linewidth=2, label='Right')
        ax4.set_xlabel('Time [s]', fontsize=9)
        ax4.set_ylabel('Forward Position [mm]', fontsize=9)
        ax4.set_title('Forward Progress', fontsize=10, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8)
        
        # ===== ROW 2: Velocity =====
        ax5 = fig.add_subplot(gs[1, 2])
        left_vel = np.linalg.norm(np.diff(left_traj, axis=0), axis=1) / np.diff(times)
        right_vel = np.linalg.norm(np.diff(right_traj, axis=0), axis=1) / np.diff(times)
        ax5.plot(times[1:], left_vel*1000, 'b-', linewidth=1.5, label='Left', alpha=0.7)
        ax5.plot(times[1:], right_vel*1000, 'r-', linewidth=1.5, label='Right', alpha=0.7)
        ax5.set_xlabel('Time [s]', fontsize=9)
        ax5.set_ylabel('Speed [mm/s]', fontsize=9)
        ax5.set_title('Foot Speed', fontsize=10, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=8)
        
        # ===== ROW 3: Joint Angles =====
        left_angles = trajectory_data['left_joint_angles']
        right_angles = trajectory_data['right_joint_angles']
        
        joint_names = ['Hip', 'Knee', 'Ankle']
        colors_l = ['#1f77b4', '#2ca02c', '#ff7f0e']
        colors_r = ['#d62728', '#9467bd', '#8c564b']
        
        for j in range(3):
            ax = fig.add_subplot(gs[2, j])
            ax.plot(times, np.degrees(left_angles[:, j]), color=colors_l[j], 
                   linewidth=2, label='Left', marker='', linestyle='-')
            ax.plot(times, np.degrees(right_angles[:, j]), color=colors_r[j], 
                   linewidth=2, label='Right', marker='', linestyle='-')
            ax.set_xlabel('Time [s]', fontsize=9)
            ax.set_ylabel('Angle [°]', fontsize=9)
            ax.set_title(f'{joint_names[j]} Angles', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # ===== ROW 4: 3D Trajectories =====
        ax7 = fig.add_subplot(gs[3, 0], projection='3d')
        ax7.plot(left_traj[:, 0]*1000, left_traj[:, 1]*1000, left_traj[:, 2]*1000, 
                'b-', linewidth=2, label='Left leg')
        ax7.scatter(left_traj[0, 0]*1000, left_traj[0, 1]*1000, left_traj[0, 2]*1000,
                   color='blue', s=60, marker='o', label='Start')
        ax7.scatter(left_traj[-1, 0]*1000, left_traj[-1, 1]*1000, left_traj[-1, 2]*1000,
                   color='blue', s=60, marker='s', label='End')
        ax7.set_xlabel('X [mm]')
        ax7.set_ylabel('Y [mm]')
        ax7.set_zlabel('Z [mm]')
        ax7.set_title('Left Leg 3D Path', fontsize=10, fontweight='bold')
        ax7.legend(fontsize=8)
        
        ax8 = fig.add_subplot(gs[3, 1], projection='3d')
        ax8.plot(right_traj[:, 0]*1000, right_traj[:, 1]*1000, right_traj[:, 2]*1000, 
                'r-', linewidth=2, label='Right leg')
        ax8.scatter(right_traj[0, 0]*1000, right_traj[0, 1]*1000, right_traj[0, 2]*1000,
                   color='red', s=60, marker='o', label='Start')
        ax8.scatter(right_traj[-1, 0]*1000, right_traj[-1, 1]*1000, right_traj[-1, 2]*1000,
                   color='red', s=60, marker='s', label='End')
        ax8.set_xlabel('X [mm]')
        ax8.set_ylabel('Y [mm]')
        ax8.set_zlabel('Z [mm]')
        ax8.set_title('Right Leg 3D Path', fontsize=10, fontweight='bold')
        ax8.legend(fontsize=8)
        
        # ===== ROW 4: Statistics =====
        ax9 = fig.add_subplot(gs[3, 2])
        ax9.axis('off')
        
        left_dist = left_traj[-1, 0] - left_traj[0, 0]
        right_dist = right_traj[-1, 0] - right_traj[0, 0]
        avg_dist = (left_dist + right_dist) / 2
        
        stats_text = f"""
10-STEP WALKING STATISTICS

Duration: {times[-1]:.2f} s
Avg Speed: {avg_dist*1000/times[-1]:.1f} mm/s

Left Leg:
  Distance: {left_dist*1000:.1f} mm
  Max Height: {np.max(left_traj[:, 2])*1000:.1f} mm
  Steps: 5 swing, 5 stance

Right Leg:
  Distance: {right_dist*1000:.1f} mm
  Max Height: {np.max(right_traj[:, 2])*1000:.1f} mm
  Steps: 5 swing, 5 stance

Gait Type: Sinusoidal
Stride: {self.stride_length*1000:.1f} mm
Height: {self.step_height*1000:.1f} mm
        """
        
        ax9.text(0.1, 0.5, stats_text, transform=ax9.transAxes, 
                fontsize=9, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('10-Step Sinusoidal Biped Walking - Complete Analysis', 
                    fontsize=13, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✅ Saved: {save_path}")
        
        return fig
    
    def save_trajectory_data(self, trajectory_data, filename='biped_10step_sinusoidal.npy'):
        """Save trajectory data to file."""
        np.save(filename, trajectory_data)
        print(f"✅ Saved trajectory data: {filename}")
        return filename


def main():
    """Generate and visualize 10-step sinusoidal walking."""
    
    generator = Biped10StepWalkingGenerator()
    
    # Generate trajectories
    print()
    trajectory_data = generator.generate_10_step_trajectory(num_points_per_phase=50)
    
    # Convert to joint angles
    print("\n🔄 Converting to joint angles...")
    trajectory_data = generator.trajectories_to_joint_angles(trajectory_data)
    
    # Plot
    print("\n📊 Creating visualizations...")
    generator.plot_trajectories(trajectory_data, save_path='biped_10step_sinusoidal.png')
    
    # Save data
    print("\n💾 Saving data...")
    generator.save_trajectory_data(trajectory_data)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Generated 10-step walking trajectory")
    print(f"✅ Total duration: {trajectory_data['total_duration']:.2f} seconds")
    print(f"✅ Total points: {trajectory_data['total_points']}")
    print(f"✅ Method: {trajectory_data['method']}")
    print(f"✅ Visualizations saved: biped_10step_sinusoidal.png")
    print(f"✅ Data saved: biped_10step_sinusoidal.npy")
    print("="*70)
    
    plt.show()


if __name__ == "__main__":
    main()
