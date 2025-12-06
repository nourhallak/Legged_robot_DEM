#!/usr/bin/env python3
"""
Trajectory Analysis and Visualization

Analyzes walking trajectories and generates detailed reports.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class TrajectoryAnalyzer:
    """Analyzes biped walking trajectories."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.base_traj = None
        self.foot1_traj = None
        self.foot2_traj = None
        self.q_left = None
        self.q_right = None
    
    def load_data(self):
        """Load trajectory data."""
        print("Loading trajectory data...")
        
        try:
            self.base_traj = np.load("base_trajectory.npy")
            self.foot1_traj = np.load("foot1_trajectory.npy")
            self.foot2_traj = np.load("foot2_trajectory.npy")
            print("✓ End-effector trajectories loaded")
        except FileNotFoundError as e:
            print(f"✗ Error loading trajectories: {e}")
            return False
        
        try:
            self.q_left = np.load("left_leg_angles.npy")
            self.q_right = np.load("right_leg_angles.npy")
            print("✓ Joint angle trajectories loaded")
        except FileNotFoundError:
            print("  (Joint angles not available)")
        
        return True
    
    def compute_statistics(self):
        """Compute trajectory statistics."""
        print("\n" + "="*80)
        print("TRAJECTORY STATISTICS")
        print("="*80)
        
        print("\nBase (Hip) Trajectory:")
        print(f"  X: {self.base_traj[:, 0].min()*1000:.2f} to {self.base_traj[:, 0].max()*1000:.2f} mm")
        print(f"  Y: {self.base_traj[:, 1].min()*1000:.2f} to {self.base_traj[:, 1].max()*1000:.2f} mm")
        print(f"  Z: {self.base_traj[:, 2].min()*1000:.2f} to {self.base_traj[:, 2].max()*1000:.2f} mm")
        print(f"  Total distance: {(self.base_traj[-1, 0] - self.base_traj[0, 0])*1000:.2f} mm")
        
        print("\nFoot1 (Left) Trajectory:")
        print(f"  X: {self.foot1_traj[:, 0].min()*1000:.2f} to {self.foot1_traj[:, 0].max()*1000:.2f} mm")
        print(f"  Y: {self.foot1_traj[:, 1].min()*1000:.2f} to {self.foot1_traj[:, 1].max()*1000:.2f} mm")
        print(f"  Z: {self.foot1_traj[:, 2].min()*1000:.2f} to {self.foot1_traj[:, 2].max()*1000:.2f} mm")
        
        print("\nFoot2 (Right) Trajectory:")
        print(f"  X: {self.foot2_traj[:, 0].min()*1000:.2f} to {self.foot2_traj[:, 0].max()*1000:.2f} mm")
        print(f"  Y: {self.foot2_traj[:, 1].min()*1000:.2f} to {self.foot2_traj[:, 1].max()*1000:.2f} mm")
        print(f"  Z: {self.foot2_traj[:, 2].min()*1000:.2f} to {self.foot2_traj[:, 2].max()*1000:.2f} mm")
        
        if self.q_left is not None:
            print("\nLeft Leg Joint Angles:")
            for i in range(3):
                print(f"  Joint {i}: {np.degrees(self.q_left[:, i].min()):.1f}° to {np.degrees(self.q_left[:, i].max()):.1f}°")
            
            print("\nRight Leg Joint Angles:")
            for i in range(3):
                print(f"  Joint {i}: {np.degrees(self.q_right[:, i].min()):.1f}° to {np.degrees(self.q_right[:, i].max()):.1f}°")
    
    def compute_velocities(self):
        """Compute trajectory velocities."""
        print("\n" + "="*80)
        print("VELOCITY ANALYSIS")
        print("="*80)
        
        # Finite differences
        base_vel = np.diff(self.base_traj, axis=0)
        foot1_vel = np.diff(self.foot1_traj, axis=0)
        foot2_vel = np.diff(self.foot2_traj, axis=0)
        
        base_speed = np.linalg.norm(base_vel, axis=1)
        foot1_speed = np.linalg.norm(foot1_vel, axis=1)
        foot2_speed = np.linalg.norm(foot2_vel, axis=1)
        
        print(f"\nBase Velocity:")
        print(f"  Mean: {base_speed.mean()*1000:.2f} mm/step")
        print(f"  Max: {base_speed.max()*1000:.2f} mm/step")
        print(f"  Std: {base_speed.std()*1000:.2f} mm/step")
        
        print(f"\nFoot1 Velocity:")
        print(f"  Mean: {foot1_speed.mean()*1000:.2f} mm/step")
        print(f"  Max: {foot1_speed.max()*1000:.2f} mm/step")
        
        print(f"\nFoot2 Velocity:")
        print(f"  Mean: {foot2_speed.mean()*1000:.2f} mm/step")
        print(f"  Max: {foot2_speed.max()*1000:.2f} mm/step")
        
        return base_speed, foot1_speed, foot2_speed
    
    def generate_report(self, output_file="trajectory_report.txt"):
        """Generate text report."""
        print(f"\nGenerating report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BIPED ROBOT WALKING TRAJECTORY REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("TRAJECTORY STATISTICS\n")
            f.write("-"*80 + "\n")
            
            f.write(f"Base (Hip) Trajectory:\n")
            f.write(f"  Points: {len(self.base_traj)}\n")
            f.write(f"  X range: {self.base_traj[:, 0].min()*1000:.2f} to {self.base_traj[:, 0].max()*1000:.2f} mm\n")
            f.write(f"  Y range: {self.base_traj[:, 1].min()*1000:.2f} to {self.base_traj[:, 1].max()*1000:.2f} mm\n")
            f.write(f"  Z range: {self.base_traj[:, 2].min()*1000:.2f} to {self.base_traj[:, 2].max()*1000:.2f} mm\n\n")
            
            f.write(f"Foot1 (Left) Trajectory:\n")
            f.write(f"  Points: {len(self.foot1_traj)}\n")
            f.write(f"  X range: {self.foot1_traj[:, 0].min()*1000:.2f} to {self.foot1_traj[:, 0].max()*1000:.2f} mm\n")
            f.write(f"  Y range: {self.foot1_traj[:, 1].min()*1000:.2f} to {self.foot1_traj[:, 1].max()*1000:.2f} mm\n")
            f.write(f"  Z range: {self.foot1_traj[:, 2].min()*1000:.2f} to {self.foot1_traj[:, 2].max()*1000:.2f} mm\n\n")
            
            f.write(f"Foot2 (Right) Trajectory:\n")
            f.write(f"  Points: {len(self.foot2_traj)}\n")
            f.write(f"  X range: {self.foot2_traj[:, 0].min()*1000:.2f} to {self.foot2_traj[:, 0].max()*1000:.2f} mm\n")
            f.write(f"  Y range: {self.foot2_traj[:, 1].min()*1000:.2f} to {self.foot2_traj[:, 1].max()*1000:.2f} mm\n")
            f.write(f"  Z range: {self.foot2_traj[:, 2].min()*1000:.2f} to {self.foot2_traj[:, 2].max()*1000:.2f} mm\n\n")
        
        print(f"✓ Report saved: {output_file}")
    
    def plot_detailed(self, output_file="detailed_analysis.png"):
        """Generate detailed analysis plots."""
        print(f"\nGenerating detailed plots: {output_file}")
        
        base_speed, foot1_speed, foot2_speed = self.compute_velocities()
        
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        fig.suptitle("Detailed Trajectory Analysis", fontsize=16, fontweight='bold')
        
        # Row 1: Position trajectories
        ax = axes[0, 0]
        ax.plot(self.base_traj[:, 0]*1000, label='Base')
        ax.plot(self.foot1_traj[:, 0]*1000, label='Foot1', alpha=0.7)
        ax.plot(self.foot2_traj[:, 0]*1000, label='Foot2', alpha=0.7)
        ax.set_ylabel('X Position (mm)')
        ax.set_title('X Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(self.base_traj[:, 1]*1000, label='Base')
        ax.plot(self.foot1_traj[:, 1]*1000, label='Foot1', alpha=0.7)
        ax.plot(self.foot2_traj[:, 1]*1000, label='Foot2', alpha=0.7)
        ax.set_ylabel('Y Position (mm)')
        ax.set_title('Y Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        ax.plot(self.base_traj[:, 2]*1000, label='Base')
        ax.plot(self.foot1_traj[:, 2]*1000, label='Foot1', alpha=0.7)
        ax.plot(self.foot2_traj[:, 2]*1000, label='Foot2', alpha=0.7)
        ax.axhline(210, color='brown', linestyle='--', label='Ground', alpha=0.5)
        ax.set_ylabel('Z Position (mm)')
        ax.set_title('Z Position (Height)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Row 2: Velocities
        ax = axes[1, 0]
        ax.plot(base_speed*1000, label='Base')
        ax.plot(foot1_speed*1000, label='Foot1', alpha=0.7)
        ax.plot(foot2_speed*1000, label='Foot2', alpha=0.7)
        ax.set_ylabel('Speed (mm/step)')
        ax.set_title('Velocity Magnitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Row 2 (cont): Phase information
        ax = axes[1, 1]
        foot1_z = self.foot1_traj[:, 2]
        foot2_z = self.foot2_traj[:, 2]
        foot1_flying = foot1_z > 0.211
        foot2_flying = foot2_z > 0.211
        
        ax.fill_between(range(len(foot1_flying)), 0, foot1_flying*1, alpha=0.5, label='Foot1', color='blue')
        ax.fill_between(range(len(foot2_flying)), 1, 1 + foot2_flying*1, alpha=0.5, label='Foot2', color='red')
        ax.set_ylabel('Phase')
        ax.set_title('Gait Phase (Stance/Swing)')
        ax.set_ylim(-0.1, 2.1)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Foot1', 'Foot2'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Row 2: Foot separation
        ax = axes[1, 2]
        separation = np.linalg.norm(self.foot1_traj - self.foot2_traj, axis=1)
        ax.plot(separation*1000)
        ax.set_ylabel('Distance (mm)')
        ax.set_title('Foot-to-Foot Separation')
        ax.grid(True, alpha=0.3)
        
        # Row 3: 3D views
        ax = axes[2, 0]
        ax.plot(self.base_traj[:, 0]*1000, self.base_traj[:, 2]*1000, label='Base')
        ax.plot(self.foot1_traj[:, 0]*1000, self.foot1_traj[:, 2]*1000, label='Foot1', alpha=0.7)
        ax.plot(self.foot2_traj[:, 0]*1000, self.foot2_traj[:, 2]*1000, label='Foot2', alpha=0.7)
        ax.axhline(210, color='brown', linestyle='--', alpha=0.5)
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Z Position (mm)')
        ax.set_title('Side View (XZ)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[2, 1]
        ax.plot(self.base_traj[:, 1]*1000, self.base_traj[:, 2]*1000, label='Base')
        ax.plot(self.foot1_traj[:, 1]*1000, self.foot1_traj[:, 2]*1000, label='Foot1', alpha=0.7)
        ax.plot(self.foot2_traj[:, 1]*1000, self.foot2_traj[:, 2]*1000, label='Foot2', alpha=0.7)
        ax.axhline(210, color='brown', linestyle='--', alpha=0.5)
        ax.set_xlabel('Y Position (mm)')
        ax.set_ylabel('Z Position (mm)')
        ax.set_title('Front View (YZ)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Joint angles if available
        if self.q_left is not None:
            ax = axes[2, 2]
            ax.plot(np.degrees(self.q_left), label=['Hip', 'Knee', 'Ankle'])
            ax.set_ylabel('Angle (degrees)')
            ax.set_title('Left Leg Joint Angles')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            axes[2, 2].text(0.5, 0.5, 'Joint angles\nnot available', 
                           ha='center', va='center', transform=axes[2, 2].transAxes)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Plots saved: {output_file}")


def main():
    """Run trajectory analysis."""
    
    print("="*80)
    print("TRAJECTORY ANALYSIS TOOL")
    print("="*80)
    
    analyzer = TrajectoryAnalyzer()
    
    if not analyzer.load_data():
        return
    
    analyzer.compute_statistics()
    analyzer.compute_velocities()
    analyzer.generate_report()
    analyzer.plot_detailed()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
