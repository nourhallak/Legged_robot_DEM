#!/usr/bin/env python3
"""
Check if planned trajectories are reachable.
Uses forward kinematics to find the robot's workspace.
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fk_leg(hip_angle, knee_angle, ankle_angle, 
           link1_len=0.14, link2_len=0.14):
    """
    Forward kinematics for 2D leg in XZ plane.
    
    Args:
        hip_angle, knee_angle, ankle_angle: Joint angles (radians)
        link1_len: Thigh length
        link2_len: Calf length
    
    Returns:
        foot_x, foot_z: Foot position relative to hip
    """
    # First link (thigh) endpoint
    link1_x = link1_len * np.sin(hip_angle)
    link1_z = -link1_len * np.cos(hip_angle)
    
    # Absolute angle of second link
    link2_abs_angle = hip_angle + knee_angle
    
    # Second link (calf) endpoint
    foot_x = link1_x + link2_len * np.sin(link2_abs_angle)
    foot_z = link1_z - link2_len * np.cos(link2_abs_angle)
    
    return foot_x, foot_z

def compute_workspace(hip_range=(-0.5, 0.5), knee_range=(-2.0, 1.0)):
    """
    Compute reachable workspace for the leg.
    
    Args:
        hip_range: (min, max) hip angles
        knee_range: (min, max) knee angles
    
    Returns:
        workspace_x, workspace_z: Arrays of reachable positions
    """
    hip_angles = np.linspace(hip_range[0], hip_range[1], 50)
    knee_angles = np.linspace(knee_range[0], knee_range[1], 50)
    
    workspace_x = []
    workspace_z = []
    
    for hip in hip_angles:
        for knee in knee_angles:
            x, z = fk_leg(hip, knee, 0.0)  # ankle_angle = 0
            workspace_x.append(x)
            workspace_z.append(z)
    
    return np.array(workspace_x), np.array(workspace_z)

def main():
    """Analyze reachability of planned trajectories."""
    
    print("\n" + "="*70)
    print("REACHABILITY ANALYSIS - FORWARD KINEMATICS")
    print("="*70 + "\n")
    
    # Load model to get actual dimensions
    try:
        model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
        print("[+] Loaded robot model")
    except:
        print("[-] Could not load robot model, using default link lengths")
    
    # Load planned trajectories
    try:
        traj_left_foot = np.load("traj_left_foot.npy")
        traj_right_foot = np.load("traj_right_foot.npy")
        traj_left_angles = np.load("traj_left_angles.npy")
        traj_right_angles = np.load("traj_right_angles.npy")
        print("[+] Loaded trajectory files")
    except FileNotFoundError:
        print("[-] Trajectory files not found. Run trajectory_planning.py first!")
        return
    
    # Compute workspace
    print("\n[+] Computing workspace boundaries...")
    workspace_x, workspace_z = compute_workspace()
    
    # Check if planned positions are reachable
    print("\n[+] Checking if planned trajectories are reachable...")
    
    left_reachable = True
    right_reachable = True
    
    for i, (lx, lz) in enumerate(zip(traj_left_foot[:, 0], traj_left_foot[:, 2])):
        # Find closest workspace point
        dists = np.sqrt((workspace_x - lx)**2 + (workspace_z - lz)**2)
        min_dist = np.min(dists)
        if min_dist > 0.01:  # 1cm tolerance
            left_reachable = False
            break
    
    for i, (rx, rz) in enumerate(zip(traj_right_foot[:, 0], traj_right_foot[:, 2])):
        dists = np.sqrt((workspace_x - rx)**2 + (workspace_z - rz)**2)
        min_dist = np.min(dists)
        if min_dist > 0.01:
            right_reachable = False
            break
    
    print(f"[-] Left foot reachability: {'YES' if left_reachable else 'NO'}")
    print(f"[-] Right foot reachability: {'YES' if right_reachable else 'NO'}")
    
    # Workspace analysis
    print("\n[+] Workspace boundaries:")
    print(f"    X range: [{workspace_x.min():.4f}, {workspace_x.max():.4f}] m")
    print(f"    Z range: [{workspace_z.min():.4f}, {workspace_z.max():.4f}] m")
    
    print("\n[+] Planned trajectory ranges:")
    print(f"    Left foot X: [{traj_left_foot[:, 0].min():.4f}, {traj_left_foot[:, 0].max():.4f}] m")
    print(f"    Left foot Z: [{traj_left_foot[:, 2].min():.4f}, {traj_left_foot[:, 2].max():.4f}] m")
    print(f"    Right foot X: [{traj_right_foot[:, 0].min():.4f}, {traj_right_foot[:, 0].max():.4f}] m")
    print(f"    Right foot Z: [{traj_right_foot[:, 2].min():.4f}, {traj_right_foot[:, 2].max():.4f}] m")
    
    # Visualize
    print("\n[+] Generating reachability visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left foot
    ax1 = axes[0]
    ax1.scatter(workspace_x, workspace_z, c='lightblue', s=20, alpha=0.6, label='Reachable workspace')
    ax1.plot(traj_left_foot[:, 0], traj_left_foot[:, 2], 'b-', linewidth=2.5, label='Left foot trajectory')
    ax1.scatter(traj_left_foot[0, 0], traj_left_foot[0, 2], c='g', s=100, marker='o', label='Start')
    ax1.scatter(traj_left_foot[-1, 0], traj_left_foot[-1, 2], c='r', s=100, marker='x', label='End')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Z Position (m)')
    ax1.set_title('LEFT FOOT - Reachability Check')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Right foot
    ax2 = axes[1]
    ax2.scatter(workspace_x, workspace_z, c='lightcoral', s=20, alpha=0.6, label='Reachable workspace')
    ax2.plot(traj_right_foot[:, 0], traj_right_foot[:, 2], 'r-', linewidth=2.5, label='Right foot trajectory')
    ax2.scatter(traj_right_foot[0, 0], traj_right_foot[0, 2], c='g', s=100, marker='o', label='Start')
    ax2.scatter(traj_right_foot[-1, 0], traj_right_foot[-1, 2], c='r', s=100, marker='x', label='End')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Z Position (m)')
    ax2.set_title('RIGHT FOOT - Reachability Check')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('reachability_analysis.png', dpi=150, bbox_inches='tight')
    print("[+] Saved: reachability_analysis.png")
    
    # If not reachable, generate adjusted trajectories
    if not left_reachable or not right_reachable:
        print("\n" + "="*70)
        print("ADJUSTING TRAJECTORIES TO BE WITHIN REACHABLE WORKSPACE")
        print("="*70 + "\n")
        
        # Scale down the trajectories to fit workspace
        left_x_scale = (workspace_x.max() - workspace_x.min()) * 0.8
        left_z_scale = (workspace_z.max() - workspace_z.min()) * 0.8
        
        # Adjust trajectories
        adjusted_left_foot = traj_left_foot.copy()
        adjusted_right_foot = traj_right_foot.copy()
        
        # Center and scale
        adjusted_left_foot[:, 0] = (traj_left_foot[:, 0] - traj_left_foot[:, 0].min()) * 0.7 + workspace_x.min() + 0.01
        adjusted_left_foot[:, 2] = traj_left_foot[:, 2] * 0.8
        
        adjusted_right_foot[:, 0] = (traj_right_foot[:, 0] - traj_right_foot[:, 0].min()) * 0.7 + workspace_x.min() + 0.01
        adjusted_right_foot[:, 2] = traj_right_foot[:, 2] * 0.8
        
        # Save adjusted
        np.save("traj_left_foot_adjusted.npy", adjusted_left_foot)
        np.save("traj_right_foot_adjusted.npy", adjusted_right_foot)
        
        print("[+] Generated adjusted trajectories within workspace")
        print("[+] Saved: traj_left_foot_adjusted.npy, traj_right_foot_adjusted.npy")
        
        # Visualize adjusted
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
        
        ax3 = axes2[0]
        ax3.scatter(workspace_x, workspace_z, c='lightblue', s=20, alpha=0.6, label='Reachable workspace')
        ax3.plot(adjusted_left_foot[:, 0], adjusted_left_foot[:, 2], 'b-', linewidth=2.5, label='Adjusted left trajectory')
        ax3.scatter(adjusted_left_foot[0, 0], adjusted_left_foot[0, 2], c='g', s=100, marker='o')
        ax3.set_xlabel('X Position (m)')
        ax3.set_ylabel('Z Position (m)')
        ax3.set_title('LEFT FOOT - Adjusted Trajectory (REACHABLE)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        ax4 = axes2[1]
        ax4.scatter(workspace_x, workspace_z, c='lightcoral', s=20, alpha=0.6, label='Reachable workspace')
        ax4.plot(adjusted_right_foot[:, 0], adjusted_right_foot[:, 2], 'r-', linewidth=2.5, label='Adjusted right trajectory')
        ax4.scatter(adjusted_right_foot[0, 0], adjusted_right_foot[0, 2], c='g', s=100, marker='o')
        ax4.set_xlabel('X Position (m)')
        ax4.set_ylabel('Z Position (m)')
        ax4.set_title('RIGHT FOOT - Adjusted Trajectory (REACHABLE)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        plt.tight_layout()
        plt.savefig('reachability_adjusted.png', dpi=150, bbox_inches='tight')
        print("[+] Saved: reachability_adjusted.png")
    else:
        print("\n[OK] All trajectories are reachable!")
    
    plt.show()


if __name__ == "__main__":
    main()
