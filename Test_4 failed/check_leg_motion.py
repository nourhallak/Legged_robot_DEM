#!/usr/bin/env python3
"""
Check IK joint angles for both legs
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
data = mujoco.MjData(model)
ik_solutions = np.load("joint_solutions_ik.npy")

print("="*80)
print("JOINT ANGLE ANALYSIS")
print("="*80)

# Joint mapping: 
# Leg1 (left): joints 3, 4, 5
# Leg2 (right): joints 6, 7, 8

print("\nJoint assignment in model:")
for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    print(f"  Joint {j}: {name}")

print(f"\nAnalyzing {len(ik_solutions)} IK solutions")
print("Legend: Leg1 = joints [3,4,5] (left), Leg2 = joints [6,7,8] (right)")

# Sample frames across the gait cycle
print("\n" + "="*80)
print("SAMPLE JOINT ANGLES ACROSS GAIT")
print("="*80)

foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

for t in [0, 25, 50, 75]:
    qpos = ik_solutions[t]
    
    foot1_z = foot1_traj[t, 2]
    foot2_z = foot2_traj[t, 2]
    
    foot1_phase = "SWING" if foot1_z > 0.431 else "STANCE"
    foot2_phase = "SWING" if foot2_z > 0.431 else "STANCE"
    
    print(f"\nStep {t} - Foot1: {foot1_phase}, Foot2: {foot2_phase}")
    print(f"  Leg1 angles (deg): [{np.degrees(qpos[3]):.1f}, {np.degrees(qpos[4]):.1f}, {np.degrees(qpos[5]):.1f}]")
    print(f"  Leg2 angles (deg): [{np.degrees(qpos[6]):.1f}, {np.degrees(qpos[7]):.1f}, {np.degrees(qpos[8]):.1f}]")
    
    # Check range of motion
    print(f"  Leg1 motion: ", end="")
    if abs(qpos[3]) < 0.01 and abs(qpos[4]) < 0.01 and abs(qpos[5]) < 0.01:
        print("MINIMAL MOVEMENT")
    else:
        print("MOVING")
    
    print(f"  Leg2 motion: ", end="")
    if abs(qpos[6]) < 0.01 and abs(qpos[7]) < 0.01 and abs(qpos[8]) < 0.01:
        print("MINIMAL MOVEMENT")
    else:
        print("MOVING")

# Check overall statistics
print("\n" + "="*80)
print("JOINT ANGLE STATISTICS")
print("="*80)

leg1_angles = ik_solutions[:, [3, 4, 5]]
leg2_angles = ik_solutions[:, [6, 7, 8]]

print(f"\nLeg1 (Left) - Joints 3,4,5:")
print(f"  Joint 3: min={np.degrees(leg1_angles[:, 0].min()):.1f}°, max={np.degrees(leg1_angles[:, 0].max()):.1f}°, range={np.degrees(leg1_angles[:, 0].max() - leg1_angles[:, 0].min()):.1f}°")
print(f"  Joint 4: min={np.degrees(leg1_angles[:, 1].min()):.1f}°, max={np.degrees(leg1_angles[:, 1].max()):.1f}°, range={np.degrees(leg1_angles[:, 1].max() - leg1_angles[:, 1].min()):.1f}°")
print(f"  Joint 5: min={np.degrees(leg1_angles[:, 2].min()):.1f}°, max={np.degrees(leg1_angles[:, 2].max()):.1f}°, range={np.degrees(leg1_angles[:, 2].max() - leg1_angles[:, 2].min()):.1f}°")

print(f"\nLeg2 (Right) - Joints 6,7,8:")
print(f"  Joint 6: min={np.degrees(leg2_angles[:, 0].min()):.1f}°, max={np.degrees(leg2_angles[:, 0].max()):.1f}°, range={np.degrees(leg2_angles[:, 0].max() - leg2_angles[:, 0].min()):.1f}°")
print(f"  Joint 7: min={np.degrees(leg2_angles[:, 1].min()):.1f}°, max={np.degrees(leg2_angles[:, 1].max()):.1f}°, range={np.degrees(leg2_angles[:, 1].max() - leg2_angles[:, 1].min()):.1f}°")
print(f"  Joint 8: min={np.degrees(leg2_angles[:, 2].min()):.1f}°, max={np.degrees(leg2_angles[:, 2].max()):.1f}°, range={np.degrees(leg2_angles[:, 2].max() - leg2_angles[:, 2].min()):.1f}°")

# Check if one leg is stuck
leg1_std = leg1_angles.std(axis=0)
leg2_std = leg2_angles.std(axis=0)

print(f"\nJoint angle variability (standard deviation):")
print(f"  Leg1: [{np.degrees(leg1_std[0]):.2f}°, {np.degrees(leg1_std[1]):.2f}°, {np.degrees(leg1_std[2]):.2f}°]")
print(f"  Leg2: [{np.degrees(leg2_std[0]):.2f}°, {np.degrees(leg2_std[1]):.2f}°, {np.degrees(leg2_std[2]):.2f}°]")
