#!/usr/bin/env python3
"""
Visualize walking motion with the new realistic trajectories.
Shows how the robot walks with IK-computed joint angles.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    print("\n" + "="*80)
    print("REALISTIC WALKING MOTION VISUALIZATION")
    print("="*80 + "\n")
    
    # Load trajectory data
    try:
        times = np.load("traj_times.npy")
        left_foot = np.load("traj_left_foot.npy")
        right_foot = np.load("traj_right_foot.npy")
        base_pos = np.load("traj_base_pos.npy")
        left_angles = np.load("traj_left_angles.npy")
        right_angles = np.load("traj_right_angles.npy")
        print("[+] Loaded all trajectory files")
        print(f"[+] Duration: {times[-1]:.1f}s")
        print(f"[+] Total distance: {left_foot[-1, 0]:.3f}m ({left_foot[-1, 0]*100:.1f}cm)")
    except FileNotFoundError as e:
        print(f"[-] Missing file: {e}")
        return
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 14))
    
    # 1. 3D Full Trajectory
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    ax1.plot(left_foot[:, 0], left_foot[:, 1], left_foot[:, 2], 'b-', linewidth=2.5, label='Left foot', alpha=0.8)
    ax1.plot(right_foot[:, 0], right_foot[:, 1], right_foot[:, 2], 'r-', linewidth=2.5, label='Right foot', alpha=0.8)
    ax1.plot(base_pos[:, 0], base_pos[:, 1], base_pos[:, 2], 'k--', linewidth=2, label='Base COM', alpha=0.7)
    ax1.scatter(left_foot[0, 0], left_foot[0, 1], left_foot[0, 2], c='b', s=150, marker='o', zorder=5, label='Start')
    ax1.scatter(left_foot[-1, 0], left_foot[-1, 1], left_foot[-1, 2], c='b', s=150, marker='x', zorder=5, label='End')
    ax1.set_xlabel('X [m]', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Y [m]', fontsize=10, fontweight='bold')
    ax1.set_zlabel('Z [m]', fontsize=10, fontweight='bold')
    ax1.set_title('3D Walking Trajectory', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Side View (XZ Plane) - Most Important
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(left_foot[:, 0], left_foot[:, 2], 'b-', linewidth=3, label='Left foot', alpha=0.8)
    ax2.plot(right_foot[:, 0], right_foot[:, 2], 'r-', linewidth=3, label='Right foot', alpha=0.8)
    ax2.plot(base_pos[:, 0], base_pos[:, 2], 'k--', linewidth=2.5, label='Base COM', alpha=0.7)
    ax2.axhline(y=0.485, color='brown', linestyle='-', linewidth=2, alpha=0.5, label='Ground')
    ax2.scatter(left_foot[0, 0], left_foot[0, 2], c='b', s=200, marker='o', zorder=5, edgecolors='black', linewidth=2)
    ax2.scatter(left_foot[-1, 0], left_foot[-1, 2], c='b', s=200, marker='x', zorder=5, linewidth=3)
    ax2.scatter(right_foot[0, 0], right_foot[0, 2], c='r', s=200, marker='o', zorder=5, edgecolors='black', linewidth=2)
    ax2.scatter(right_foot[-1, 0], right_foot[-1, 2], c='r', s=200, marker='x', zorder=5, linewidth=3)
    ax2.set_xlabel('Forward Distance - X [m]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Height - Z [m]', fontsize=11, fontweight='bold')
    ax2.set_title('SIDE VIEW: Robot Walking Profile', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 3. Top View (XY Plane)
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(left_foot[:, 0], left_foot[:, 1], 'b-', linewidth=2.5, label='Left foot')
    ax3.plot(right_foot[:, 0], right_foot[:, 1], 'r-', linewidth=2.5, label='Right foot')
    ax3.plot(base_pos[:, 0], base_pos[:, 1], 'k--', linewidth=2, label='Base COM', alpha=0.7)
    ax3.scatter([left_foot[0, 0]], [left_foot[0, 1]], c='b', s=150, marker='o', zorder=5)
    ax3.scatter([right_foot[0, 0]], [right_foot[0, 1]], c='r', s=150, marker='o', zorder=5)
    ax3.set_xlabel('Forward Distance - X [m]', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Lateral Distance - Y [m]', fontsize=11, fontweight='bold')
    ax3.set_title('TOP VIEW: Lateral Stability', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. X Position vs Time
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(times, left_foot[:, 0]*100, 'b-', linewidth=2.5, label='Left foot', alpha=0.8)
    ax4.plot(times, right_foot[:, 0]*100, 'r-', linewidth=2.5, label='Right foot', alpha=0.8)
    ax4.plot(times, base_pos[:, 0]*100, 'k--', linewidth=2, label='Base COM', alpha=0.7)
    ax4.fill_between(times, left_foot[:, 0]*100, right_foot[:, 0]*100, alpha=0.1, color='purple')
    ax4.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Forward Position [cm]', fontsize=11, fontweight='bold')
    ax4.set_title('Forward Progress', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Z Position vs Time (Height)
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(times, left_foot[:, 2]*100, 'b-', linewidth=2.5, label='Left foot', alpha=0.8)
    ax5.plot(times, right_foot[:, 2]*100, 'r-', linewidth=2.5, label='Right foot', alpha=0.8)
    ax5.axhline(y=48.5, color='brown', linestyle='-', linewidth=2, alpha=0.5, label='Ground')
    ax5.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Height [cm]', fontsize=11, fontweight='bold')
    ax5.set_title('Swing Phases - Foot Height', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Y Position vs Time (Lateral Sway)
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(times, left_foot[:, 1]*1000, 'b-', linewidth=2.5, label='Left foot', alpha=0.8)
    ax6.plot(times, right_foot[:, 1]*1000, 'r-', linewidth=2.5, label='Right foot', alpha=0.8)
    ax6.plot(times, base_pos[:, 1]*1000, 'k--', linewidth=2, label='Base COM', alpha=0.7)
    ax6.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Lateral Position [mm]', fontsize=11, fontweight='bold')
    ax6.set_title('Lateral Stability', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # 7. Left Leg Joint Angles
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.plot(times, left_angles[:, 0], linewidth=2.5, label='Hip (abduction)', alpha=0.8, color='#1f77b4')
    ax7.plot(times, left_angles[:, 1], linewidth=2.5, label='Knee (flexion)', alpha=0.8, color='#ff7f0e')
    ax7.plot(times, left_angles[:, 2], linewidth=2.5, label='Ankle', alpha=0.8, color='#2ca02c')
    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax7.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Angle [rad]', fontsize=11, fontweight='bold')
    ax7.set_title('LEFT LEG Joint Angles', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # 8. Right Leg Joint Angles
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.plot(times, right_angles[:, 0], linewidth=2.5, label='Hip (abduction)', alpha=0.8, color='#1f77b4')
    ax8.plot(times, right_angles[:, 1], linewidth=2.5, label='Knee (flexion)', alpha=0.8, color='#ff7f0e')
    ax8.plot(times, right_angles[:, 2], linewidth=2.5, label='Ankle', alpha=0.8, color='#2ca02c')
    ax8.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax8.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Angle [rad]', fontsize=11, fontweight='bold')
    ax8.set_title('RIGHT LEG Joint Angles', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # 9. Gait Phase Indicators
    ax9 = fig.add_subplot(3, 3, 9)
    gait_phase = np.fmod(times, 2.0) / 2.0  # 2-second gait period
    left_phase = np.where(gait_phase < 0.5, 1, 0)
    right_phase = np.where(gait_phase >= 0.5, 1, 0)
    ax9.fill_between(times, 0, left_phase, alpha=0.3, color='blue', label='Left foot swing')
    ax9.fill_between(times, 0, right_phase, alpha=0.3, color='red', label='Right foot swing')
    ax9.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Swing Phase', fontsize=11, fontweight='bold')
    ax9.set_title('Gait Phases (Swing = 1, Stance = 0)', fontsize=12, fontweight='bold')
    ax9.set_ylim([-0.1, 1.1])
    ax9.legend(fontsize=10)
    ax9.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('BIPEDAL ROBOT WALKING WITH REALISTIC IK TRAJECTORIES', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('walking_visualization_realistic.png', dpi=150, bbox_inches='tight')
    print("\n[+] Visualization saved to 'walking_visualization_realistic.png'")
    
    # Print gait statistics
    print("\n" + "="*80)
    print("WALKING GAIT STATISTICS")
    print("="*80)
    print(f"  Total duration: {times[-1]:.1f}s")
    print(f"  Total distance traveled: {left_foot[-1, 0]*100:.1f} cm")
    print(f"  Walking speed: {(left_foot[-1, 0]/times[-1])*100:.1f} cm/s")
    print(f"  Gait period: 2.0 s (0.5 Hz)")
    print(f"  Number of complete gait cycles: {int(times[-1]/2)}")
    print(f"  Step length: {0.01*100:.1f} cm")
    print(f"  Step height: {(left_foot[:, 2].max() - 0.485)*100:.1f} cm")
    print(f"  Max forward reach: {left_foot[:, 0].max()*100:.1f} cm")
    print(f"  Lateral stability range: {(left_foot[:, 1].max() - left_foot[:, 1].min())*1000:.1f} mm")
    print(f"  Base COM height: {base_pos[0, 2]*100:.1f} cm (constant)")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
