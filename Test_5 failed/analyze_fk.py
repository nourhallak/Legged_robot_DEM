#!/usr/bin/env python3
"""
Forward Kinematics Analysis - Workspace Exploration

Analyzes the reachable workspace of the biped robot using FK.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco

def analyze_workspace():
    """Analyze reachable workspace using forward kinematics."""
    
    print("\n" + "="*80)
    print("FORWARD KINEMATICS - WORKSPACE ANALYSIS")
    print("="*80)
    
    # Load model
    model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
    data = mujoco.MjData(model)
    
    # Joint ranges to explore
    n_samples = 8  # Samples per joint
    joint_ranges = [
        np.linspace(-np.pi/2, np.pi/2, n_samples),    # Hip
        np.linspace(-np.pi/2, 0, n_samples),           # Knee  
        np.linspace(0, np.pi/2, n_samples),            # Ankle
    ]
    
    print(f"\nScanning workspace...")
    print(f"  Samples per joint: {n_samples}")
    print(f"  Total configurations: {n_samples**3}")
    
    # Collect all reachable positions
    left_positions = []
    right_positions = []
    
    sample_count = 0
    for hip in joint_ranges[0]:
        for knee in joint_ranges[1]:
            for ankle in joint_ranges[2]:
                # Set left leg
                data.qpos[3:6] = [hip, knee, ankle]
                # Mirror for right leg
                data.qpos[6:9] = [hip, knee, ankle]
                
                mujoco.mj_forward(model, data)
                
                left_pos = data.site_xpos[0].copy()
                right_pos = data.site_xpos[1].copy()
                
                left_positions.append(left_pos)
                right_positions.append(right_pos)
                
                sample_count += 1
    
    left_positions = np.array(left_positions)
    right_positions = np.array(right_positions)
    
    print(f"✓ Scanned {sample_count} configurations")
    
    # Analyze workspace
    print("\n" + "-"*80)
    print("WORKSPACE STATISTICS")
    print("-"*80)
    
    print("\nLeft Foot Reachable Space:")
    print(f"  X: {left_positions[:, 0].min()*1000:7.2f} to {left_positions[:, 0].max()*1000:7.2f} mm")
    print(f"  Y: {left_positions[:, 1].min()*1000:7.2f} to {left_positions[:, 1].max()*1000:7.2f} mm")
    print(f"  Z: {left_positions[:, 2].min()*1000:7.2f} to {left_positions[:, 2].max()*1000:7.2f} mm")
    
    print("\nRight Foot Reachable Space:")
    print(f"  X: {right_positions[:, 0].min()*1000:7.2f} to {right_positions[:, 0].max()*1000:7.2f} mm")
    print(f"  Y: {right_positions[:, 1].min()*1000:7.2f} to {right_positions[:, 1].max()*1000:7.2f} mm")
    print(f"  Z: {right_positions[:, 2].min()*1000:7.2f} to {right_positions[:, 2].max()*1000:7.2f} mm")
    
    # Load trajectories and check if reachable
    print("\n" + "-"*80)
    print("TRAJECTORY REACHABILITY CHECK")
    print("-"*80)
    
    foot1_traj = np.load("foot1_trajectory.npy")
    foot2_traj = np.load("foot2_trajectory.npy")
    
    # Check if trajectory points are within workspace
    left_x_in = np.all((foot1_traj[:, 0] >= left_positions[:, 0].min()) & 
                       (foot1_traj[:, 0] <= left_positions[:, 0].max()))
    left_y_in = np.all((foot1_traj[:, 1] >= left_positions[:, 1].min()) & 
                       (foot1_traj[:, 1] <= left_positions[:, 1].max()))
    left_z_in = np.all((foot1_traj[:, 2] >= left_positions[:, 2].min()) & 
                       (foot1_traj[:, 2] <= left_positions[:, 2].max()))
    
    print(f"\nLeft foot trajectory reachability:")
    print(f"  X range: {foot1_traj[:, 0].min()*1000:.2f} - {foot1_traj[:, 0].max()*1000:.2f} mm", end="")
    print(f"  {'✓ REACHABLE' if left_x_in else '✗ OUT OF RANGE'}")
    print(f"  Y range: {foot1_traj[:, 1].min()*1000:.2f} - {foot1_traj[:, 1].max()*1000:.2f} mm", end="")
    print(f"  {'✓ REACHABLE' if left_y_in else '✗ OUT OF RANGE'}")
    print(f"  Z range: {foot1_traj[:, 2].min()*1000:.2f} - {foot1_traj[:, 2].max()*1000:.2f} mm", end="")
    print(f"  {'✓ REACHABLE' if left_z_in else '✗ OUT OF RANGE'}")
    
    right_x_in = np.all((foot2_traj[:, 0] >= right_positions[:, 0].min()) & 
                        (foot2_traj[:, 0] <= right_positions[:, 0].max()))
    right_y_in = np.all((foot2_traj[:, 1] >= right_positions[:, 1].min()) & 
                        (foot2_traj[:, 1] <= right_positions[:, 1].max()))
    right_z_in = np.all((foot2_traj[:, 2] >= right_positions[:, 2].min()) & 
                        (foot2_traj[:, 2] <= right_positions[:, 2].max()))
    
    print(f"\nRight foot trajectory reachability:")
    print(f"  X range: {foot2_traj[:, 0].min()*1000:.2f} - {foot2_traj[:, 0].max()*1000:.2f} mm", end="")
    print(f"  {'✓ REACHABLE' if right_x_in else '✗ OUT OF RANGE'}")
    print(f"  Y range: {foot2_traj[:, 1].min()*1000:.2f} - {foot2_traj[:, 1].max()*1000:.2f} mm", end="")
    print(f"  {'✓ REACHABLE' if right_y_in else '✗ OUT OF RANGE'}")
    print(f"  Z range: {foot2_traj[:, 2].min()*1000:.2f} - {foot2_traj[:, 2].max()*1000:.2f} mm", end="")
    print(f"  {'✓ REACHABLE' if right_z_in else '✗ OUT OF RANGE'}")
    
    # Save workspace data
    print("\n" + "-"*80)
    print("SAVING WORKSPACE DATA")
    print("-"*80)
    
    np.save("left_workspace.npy", left_positions)
    np.save("right_workspace.npy", right_positions)
    print("✓ left_workspace.npy")
    print("✓ right_workspace.npy")
    
    # Generate visualization
    print("\nGenerating workspace visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(left_positions[:, 0]*1000, left_positions[:, 1]*1000, left_positions[:, 2]*1000, 
               c='blue', s=20, alpha=0.6, label='Left foot reachable')
    ax1.scatter(foot1_traj[:, 0]*1000, foot1_traj[:, 1]*1000, foot1_traj[:, 2]*1000, 
               c='red', s=30, alpha=0.8, label='Left foot trajectory')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('Left Foot - Workspace vs Trajectory (3D)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # XY plane
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(left_positions[:, 0]*1000, left_positions[:, 1]*1000, c='blue', s=20, alpha=0.6, label='Reachable')
    ax2.scatter(foot1_traj[:, 0]*1000, foot1_traj[:, 1]*1000, c='red', s=30, alpha=0.8, label='Trajectory')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('XY Plane (Top View)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # XZ plane
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(left_positions[:, 0]*1000, left_positions[:, 2]*1000, c='blue', s=20, alpha=0.6, label='Reachable')
    ax3.scatter(foot1_traj[:, 0]*1000, foot1_traj[:, 2]*1000, c='red', s=30, alpha=0.8, label='Trajectory')
    ax3.axhline(210, color='brown', linestyle='--', alpha=0.5, label='Ground')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('XZ Plane (Side View)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Right foot workspace
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.scatter(right_positions[:, 0]*1000, right_positions[:, 1]*1000, right_positions[:, 2]*1000,
               c='green', s=20, alpha=0.6, label='Right foot reachable')
    ax4.scatter(foot2_traj[:, 0]*1000, foot2_traj[:, 1]*1000, foot2_traj[:, 2]*1000,
               c='red', s=30, alpha=0.8, label='Right foot trajectory')
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_zlabel('Z (mm)')
    ax4.set_title('Right Foot - Workspace vs Trajectory (3D)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # YZ plane
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(left_positions[:, 1]*1000, left_positions[:, 2]*1000, c='blue', s=20, alpha=0.6, label='Left reachable')
    ax5.scatter(right_positions[:, 1]*1000, right_positions[:, 2]*1000, c='green', s=20, alpha=0.6, label='Right reachable')
    ax5.scatter(foot1_traj[:, 1]*1000, foot1_traj[:, 2]*1000, c='red', s=30, alpha=0.8, label='Trajectory')
    ax5.axhline(210, color='brown', linestyle='--', alpha=0.5, label='Ground')
    ax5.set_xlabel('Y (mm)')
    ax5.set_ylabel('Z (mm)')
    ax5.set_title('YZ Plane (Front View)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Volume comparison
    ax6 = fig.add_subplot(2, 3, 6)
    left_vol = (left_positions[:, 0].max() - left_positions[:, 0].min()) * \
              (left_positions[:, 1].max() - left_positions[:, 1].min()) * \
              (left_positions[:, 2].max() - left_positions[:, 2].min()) * 1e9  # Convert to mm³
    
    traj_vol = (foot1_traj[:, 0].max() - foot1_traj[:, 0].min()) * \
              (foot1_traj[:, 1].max() - foot1_traj[:, 1].min()) * \
              (foot1_traj[:, 2].max() - foot1_traj[:, 2].min()) * 1e9
    
    ax6.bar(['Workspace', 'Trajectory'], [left_vol, traj_vol], color=['blue', 'red'], alpha=0.7)
    ax6.set_ylabel('Volume (mm³)')
    ax6.set_title('Bounding Box Volume Comparison')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("workspace_analysis.png", dpi=150, bbox_inches='tight')
    print("✓ workspace_analysis.png")
    
    print("\n" + "="*80)
    print("FORWARD KINEMATICS ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    return left_positions, right_positions


if __name__ == "__main__":
    analyze_workspace()
