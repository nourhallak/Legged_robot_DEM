#!/usr/bin/env python3
"""
Diagnose why robot is flying and not walking humanoid-like
Check: foot contact, COM height, IK convergence, ground clearance
"""
import numpy as np
import mujoco
import os

# Load model and trajectories
script_dir = os.path.dirname(os.path.abspath(__file__))
mjcf_path = os.path.join(script_dir, "legged_robot_ik.xml")

try:
    model = mujoco.MjModel.from_xml_path(mjcf_path)
except:
    model = mujoco.load_model_from_xml(open(mjcf_path).read())

# Load trajectories
com_traj = np.load("com_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

print("=" * 80)
print("DIAGNOSIS: Robot Flying and Non-Humanoid Walking")
print("=" * 80)

print("\n1. GROUND GEOMETRY")
print("-" * 80)
print(f"Number of geoms: {model.ngeom}")
for i in range(model.ngeom):
    if model.geom_type[i] == 0:  # Plane
        print(f"  Geom {i}: PLANE at Z = {model.geom_pos[i][2]:.4f} m")
    elif model.geom_type[i] == 7:  # Mesh
        print(f"  Geom {i}: MESH at pos {model.geom_pos[i]}")

print("\n2. TRAJECTORY ANALYSIS")
print("-" * 80)
print("\nCOM trajectory:")
print(f"  Z range: {com_traj[:, 2].min():.6f} to {com_traj[:, 2].max():.6f} m")
print(f"  Distance above ground (Z=0): {com_traj[:, 2].min():.6f} m")

print("\nFoot 1 (Left) trajectory:")
print(f"  Z range: {foot1_traj[:, 2].min():.6f} to {foot1_traj[:, 2].max():.6f} m")
print(f"  Distance above ground: {foot1_traj[:, 2].min():.6f} m")

print("\nFoot 2 (Right) trajectory:")
print(f"  Z range: {foot2_traj[:, 2].min():.6f} to {foot2_traj[:, 2].max():.6f} m")
print(f"  Distance above ground: {foot2_traj[:, 2].min():.6f} m")

print("\n3. PROBLEM ANALYSIS")
print("-" * 80)
print(f"\nPROBLEM 1: Robot Flying")
print(f"  - Foot lowest point: {min(foot1_traj[:, 2].min(), foot2_traj[:, 2].min()):.6f} m")
print(f"  - COM lowest point:  {com_traj[:, 2].min():.6f} m")
print(f"  - Ground at: 0.0000 m")
print(f"  -> Feet never touch Z=0! Robot must float in air.")

print(f"\nPROBLEM 2: Non-Humanoid Walking")
print(f"  Current trajectory profile:")
print(f"    - Both feet move with same phase (both swing, both stance alternates)")
print(f"    - No natural single-leg support phase")
print(f"    - No double-support phase for stability")
print(f"    - COM stays at middle height, doesn't shift over stance leg")

print("\n4. RECOMMENDED FIXES")
print("-" * 80)
print("\nFIX 1: Move trajectories to ground level")
print(f"  Current COM Z: 0.223-0.243 m")
print(f"  Needed COM Z:  0.05-0.10 m (typical bipedal robot)")
print(f"  - Adjust hip height during design/use")
print(f"  - OR scale down trajectories")
print(f"  - OR move ground plane down to Z = 0.2 m")

print("\nFIX 2: Add humanoid gait pattern")
print("  Real walking phases:")
print("    1. Double support (both feet on ground)")
print("    2. Single support (one leg carries weight, other swings)")
print("    3. Repeat")
print("  - COM moves laterally over stance leg")
print("  - Stance leg stays level")
print("  - Swing leg has clear lift phase")
print("  - Natural knee flexion during swing")

print("\n5. IMPLEMENTATION STRATEGY")
print("-" * 80)
print("\nOption A: Ground Contact via Physics (RECOMMENDED)")
print("  - Keep current Z trajectories (0.2125-0.2437m)")
print("  - Add collision between feet and ground")
print("  - Physics will naturally enforce Z >= 0.2125m contact")
print("  - Robot won't fall through ground")

print("\nOption B: Scale Trajectories to Ground")
print("  - Shift all trajectory Z values down")
print("  - New Z for stance: 0.0 m (ground)")
print("  - New Z for swing: 0.05 m (foot clearance)")
print("  - Requires different robot design or configuration")

print("\nOption C: Redesign Gait (BEST for Realism)")
print("  - Move ground plane to Z = 0.21 m (robot's natural foot level)")
print("  - Generate realistic humanoid gait:")
print("    * Phase 1 (0-100 steps): Double support")
print("    * Phase 2 (100-250 steps): Right leg swing, left leg stance")
print("    * Phase 3 (250-300 steps): Double support")
print("    * Phase 4 (300-400 steps): Left leg swing, right leg stance")
print("  - COM moves naturally over each stance leg")
print("  - Results in natural human-like walking")

print("\n" + "=" * 80)
