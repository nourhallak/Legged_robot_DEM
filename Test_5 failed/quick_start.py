#!/usr/bin/env python3
"""
Quick Start Guide for Biped Robot Walking

This script runs the complete pipeline in sequence:
1. Generate walking trajectories
2. Solve inverse kinematics
3. Analyze trajectories
4. Run simulation
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a Python script and report status."""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    
    try:
        result = subprocess.run(
            [sys.executable, cmd],
            capture_output=False,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"✗ {description} timed out (>300s)")
        return False
    except Exception as e:
        print(f"✗ {description} error: {e}")
        return False


def main():
    """Run the complete walking pipeline."""
    
    print("="*80)
    print("BIPED ROBOT WALKING - QUICK START PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("  1. Generate smooth walking trajectories")
    print("  2. Solve inverse kinematics for joint angles")
    print("  3. Analyze trajectory properties")
    print("  4. Optionally run MuJoCo simulation")
    
    # Check required files
    print("\n" + "="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)
    
    required_files = [
        "generate_walking_trajectories.py",
        "legged_robot_ik.xml"
    ]
    
    missing = []
    for f in required_files:
        if Path(f).exists():
            print(f"✓ {f}")
        else:
            print(f"✗ {f} (MISSING)")
            missing.append(f)
    
    if missing:
        print(f"\n✗ Missing files: {', '.join(missing)}")
        print("Cannot proceed with pipeline.")
        return False
    
    # Run pipeline
    steps = [
        ("generate_walking_trajectories.py", "Generate Walking Trajectories"),
        ("analyze_trajectories.py", "Analyze Trajectories (Before IK)"),
    ]
    
    # Optional IK and simulation if files exist
    if Path("solve_ik.py").exists():
        steps.append(("solve_ik.py", "Solve Inverse Kinematics"))
        steps.append(("analyze_trajectories.py", "Analyze Trajectories (After IK)"))
    
    if Path("run_walking_sim.py").exists():
        steps.append(("run_walking_sim.py", "Run Walking Simulation"))
    
    # Execute pipeline
    success_count = 0
    for cmd, desc in steps:
        if run_command(cmd, desc):
            success_count += 1
        else:
            print(f"\nWarning: {desc} failed, but continuing pipeline...")
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nSuccessfully completed {success_count}/{len(steps)} steps")
    
    print("\n" + "-"*80)
    print("GENERATED FILES:")
    print("-"*80)
    
    output_files = [
        ("base_trajectory.npy", "Base (hip) position trajectory"),
        ("foot1_trajectory.npy", "Left foot position trajectory"),
        ("foot2_trajectory.npy", "Right foot position trajectory"),
        ("walking_trajectories.png", "Trajectory visualization"),
        ("left_leg_angles.npy", "Left leg joint angles (from IK)"),
        ("right_leg_angles.npy", "Right leg joint angles (from IK)"),
        ("trajectory_report.txt", "Analysis report"),
        ("detailed_analysis.png", "Detailed analysis plots"),
    ]
    
    for filename, description in output_files:
        if Path(filename).exists():
            size = Path(filename).stat().st_size
            if filename.endswith('.npy'):
                print(f"✓ {filename:30s} ({size:,} bytes) - {description}")
            else:
                print(f"✓ {filename:30s} - {description}")
        else:
            print(f"  {filename:30s} (not generated)")
    
    print("\n" + "-"*80)
    print("NEXT STEPS:")
    print("-"*80)
    print("1. Review walking_trajectories.png for trajectory visualization")
    print("2. Check trajectory_report.txt for detailed statistics")
    print("3. Review detailed_analysis.png for velocity and phase analysis")
    print("4. If IK was solved, check left_leg_angles.npy and right_leg_angles.npy")
    print("5. Run walking simulation with: python run_walking_sim.py")
    print("\n" + "="*80)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
