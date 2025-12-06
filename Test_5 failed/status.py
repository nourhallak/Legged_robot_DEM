#!/usr/bin/env python3
"""
Status and Inventory Script

Shows what files have been generated and current status.
"""

import numpy as np
from pathlib import Path
from datetime import datetime

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def check_file(filepath, description=""):
    """Check if file exists and print status."""
    p = Path(filepath)
    if p.exists():
        size = p.stat().st_size
        if filepath.endswith('.npy'):
            try:
                data = np.load(filepath)
                return f"✓ {filepath:35s} {str(data.shape):20s} ({size:,} bytes) {description}"
            except:
                return f"✓ {filepath:35s} ({size:,} bytes) {description}"
        else:
            return f"✓ {filepath:35s} ({size:,} bytes) {description}"
    else:
        return f"  {filepath:35s} (not generated)"

def main():
    """Show current status and inventory."""
    
    print("\n" + "="*80)
    print("  BIPED ROBOT WALKING - PROJECT STATUS")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Location: {Path.cwd()}")
    
    # Scripts
    print_section("AVAILABLE SCRIPTS")
    
    scripts = [
        ("generate_walking_trajectories.py", "Generate walking trajectories"),
        ("solve_ik.py", "Solve inverse kinematics"),
        ("run_walking_sim.py", "Run MuJoCo simulation"),
        ("analyze_trajectories.py", "Analyze and visualize trajectories"),
        ("quick_start.py", "Automated pipeline (all steps)"),
    ]
    
    for script, desc in scripts:
        if Path(script).exists():
            print(f"  ✓ {script:40s} - {desc}")
        else:
            print(f"    {script:40s} - {desc} (not found)")
    
    # Generated trajectory files
    print_section("TRAJECTORY DATA FILES")
    
    print(check_file("base_trajectory.npy", "Base (hip) position trajectory"))
    print(check_file("foot1_trajectory.npy", "Left foot position trajectory"))
    print(check_file("foot2_trajectory.npy", "Right foot position trajectory"))
    
    # IK solution files
    print_section("INVERSE KINEMATICS SOLUTIONS")
    
    print(check_file("left_leg_angles.npy", "Left leg joint angles"))
    print(check_file("right_leg_angles.npy", "Right leg joint angles"))
    print(check_file("left_leg_success.npy", "Left leg IK convergence flags"))
    print(check_file("right_leg_success.npy", "Right leg IK convergence flags"))
    print(check_file("left_leg_errors.npy", "Left leg position errors"))
    print(check_file("right_leg_errors.npy", "Right leg position errors"))
    
    # Analysis files
    print_section("ANALYSIS & VISUALIZATION")
    
    print(check_file("walking_trajectories.png", "Trajectory visualization"))
    print(check_file("detailed_analysis.png", "Detailed analysis plots"))
    print(check_file("trajectory_report.txt", "Analysis report"))
    
    # Documentation
    print_section("DOCUMENTATION")
    
    print(check_file("README_WALKING.md", "Walking pipeline documentation"))
    
    # Summary statistics
    print_section("TRAJECTORY STATISTICS")
    
    if Path("base_trajectory.npy").exists():
        try:
            base = np.load("base_trajectory.npy")
            foot1 = np.load("foot1_trajectory.npy")
            foot2 = np.load("foot2_trajectory.npy")
            
            print(f"\nTrajectory Points: {len(base)}")
            print(f"  Forward distance: {(base[-1, 0] - base[0, 0])*1000:.1f} mm")
            print(f"  Hip height range: {base[:, 2].min()*1000:.1f} to {base[:, 2].max()*1000:.1f} mm")
            print(f"  Foot height range: {foot1[:, 2].min()*1000:.1f} to {foot1[:, 2].max()*1000:.1f} mm")
            print(f"  Left-right spacing: {abs(foot1[0, 1] - foot2[0, 1])*1000:.1f} mm")
        except Exception as e:
            print(f"  Error reading data: {e}")
    
    if Path("left_leg_angles.npy").exists() and Path("right_leg_angles.npy").exists():
        try:
            q_left = np.load("left_leg_angles.npy")
            q_right = np.load("right_leg_angles.npy")
            
            print(f"\nJoint Angle Ranges:")
            print(f"  Left leg:  {np.degrees(q_left).min():.1f}° to {np.degrees(q_left).max():.1f}°")
            print(f"  Right leg: {np.degrees(q_right).min():.1f}° to {np.degrees(q_right).max():.1f}°")
        except Exception as e:
            print(f"  Error reading joint angles: {e}")
    
    # Next steps
    print_section("RECOMMENDED NEXT STEPS")
    
    if not Path("base_trajectory.npy").exists():
        print("  1. Generate trajectories:")
        print("     python generate_walking_trajectories.py")
    else:
        print("  ✓ Trajectories generated")
        
        if not Path("left_leg_angles.npy").exists():
            print("  2. Solve inverse kinematics:")
            print("     python solve_ik.py")
        else:
            print("  ✓ IK solved")
            
            if Path("run_walking_sim.py").exists():
                print("  3. Run simulation:")
                print("     python run_walking_sim.py")
        
        if not Path("detailed_analysis.png").exists():
            print("  • Generate detailed analysis:")
            print("     python analyze_trajectories.py")
    
    print("\n  Or run the complete pipeline:")
    print("     python quick_start.py")
    
    # Footer
    print("\n" + "="*80)
    print("  For detailed documentation, see: README_WALKING.md")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
