"""
Analyze knee joint kinematics to find appropriate joint limits
Tests different knee angles and shows what positions they can reach
"""
import mujoco
import numpy as np
import os
import re
from pathlib import Path

def load_model_with_assets():
    xml_path = Path(__file__).parent / "legged_robot_ik.xml"
    
    with open(xml_path, "r") as f:
        xml_content = f.read()
    
    asset_dir = Path(__file__).parent / "Legged_robot" / "meshes"
    
    for match in re.finditer(r'filename="([^"]*\.stl)"', xml_content):
        stl_file = match.group(1)
        full_path = asset_dir / stl_file
        
        if full_path.exists():
            xml_content = xml_content.replace(f'filename="{stl_file}"', f'filename="{full_path}"')
    
    return mujoco.MjModel.from_xml_string(xml_content)

model = load_model_with_assets()
data = mujoco.MjData(model)

print("=== KNEE JOINT KINEMATICS ANALYSIS ===\n")

com_traj = np.load("com_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

# Get site IDs
com_site_id = model.site(name='com_site').id
foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id

print("Testing knee angles for Leg 1 (left leg):")
print("Knee angle | Foot1 X  | Foot1 Z  | Reach?")
print("-----------|----------|----------|--------")

# Test a range of knee angles for leg 1
knee_angles = np.linspace(-np.pi, np.pi, 37)  # -180 to +180 degrees

leg1_results = []

for knee_angle in knee_angles:
    # Set base configuration
    data.qpos[:] = [0, 0, 0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Set leg 1 knee joint
    data.qpos[7] = knee_angle  # knee_1_joint is at index 7
    
    try:
        mujoco.mj_forward(model, data)
        foot1_pos = data.site_xpos[foot1_site_id]
        degrees = np.degrees(knee_angle)
        
        # Check if this angle can reach some reasonable foot positions
        can_reach = True
        leg1_results.append({
            'angle': knee_angle,
            'degrees': degrees,
            'foot_x': foot1_pos[0],
            'foot_z': foot1_pos[2],
            'valid': True
        })
        
        print(f"{degrees:6.1f}°    | {foot1_pos[0]:+.4f}  | {foot1_pos[2]:.4f}  | OK")
    except Exception as e:
        print(f"{np.degrees(knee_angle):6.1f}°    | ERROR  | ERROR  | FAIL")

print("\n" + "="*60)
print("\nTesting knee angles for Leg 2 (right leg):")
print("Knee angle | Foot2 X  | Foot2 Z  | Reach?")
print("-----------|----------|----------|--------")

leg2_results = []

for knee_angle in knee_angles:
    # Set base configuration
    data.qpos[:] = [0, 0, 0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Set leg 2 knee joint
    data.qpos[10] = knee_angle  # knee_2_joint is at index 10
    
    try:
        mujoco.mj_forward(model, data)
        foot2_pos = data.site_xpos[foot2_site_id]
        degrees = np.degrees(knee_angle)
        
        leg2_results.append({
            'angle': knee_angle,
            'degrees': degrees,
            'foot_x': foot2_pos[0],
            'foot_z': foot2_pos[2],
            'valid': True
        })
        
        print(f"{degrees:6.1f}°    | {foot2_pos[0]:+.4f}  | {foot2_pos[2]:.4f}  | OK")
    except Exception as e:
        print(f"{np.degrees(knee_angle):6.1f}°    | ERROR  | ERROR  | FAIL")

print("\n" + "="*60)
print("\n=== ANALYSIS FOR WALKING ===\n")

# Analyze trajectory requirements
print("Trajectory requirements:")
print(f"  Foot positions need Z ≈ 0 to 0.03m (ground to swing height)")
print(f"  Foot1 X range: {foot1_traj[:, 0].min():.4f} to {foot1_traj[:, 0].max():.4f}")
print(f"  Foot2 X range: {foot2_traj[:, 0].min():.4f} to {foot2_traj[:, 0].max():.4f}")

# Find knee angles that keep foot near ground (z ≈ 0)
print("\nKnee angles that keep foot NEAR GROUND (z < 0.01m):")
print("\nLeg 1 (index 7):")
ground_touching_leg1 = [r for r in leg1_results if r['foot_z'] < 0.01]
if ground_touching_leg1:
    angles_leg1 = [r['degrees'] for r in ground_touching_leg1]
    print(f"  Angle range: {min(angles_leg1):.1f}° to {max(angles_leg1):.1f}°")
    print(f"  Recommended limits: {np.radians(min(angles_leg1)):.3f} to {np.radians(max(angles_leg1)):.3f} rad")
else:
    print("  No angles found that keep foot on ground!")

print("\nLeg 2 (index 10):")
ground_touching_leg2 = [r for r in leg2_results if r['foot_z'] < 0.01]
if ground_touching_leg2:
    angles_leg2 = [r['degrees'] for r in ground_touching_leg2]
    print(f"  Angle range: {min(angles_leg2):.1f}° to {max(angles_leg2):.1f}°")
    print(f"  Recommended limits: {np.radians(min(angles_leg2)):.3f} to {np.radians(max(angles_leg2)):.3f} rad")
else:
    print("  No angles found that keep foot on ground!")

# Check full reachable workspace
print("\n" + "="*60)
print("\n=== FULL REACHABLE WORKSPACE ===\n")

print("Leg 1 foot position extremes:")
print(f"  X range: {min([r['foot_x'] for r in leg1_results]):.4f} to {max([r['foot_x'] for r in leg1_results]):.4f}")
print(f"  Z range: {min([r['foot_z'] for r in leg1_results]):.4f} to {max([r['foot_z'] for r in leg1_results]):.4f}")

print("\nLeg 2 foot position extremes:")
print(f"  X range: {min([r['foot_x'] for r in leg2_results]):.4f} to {max([r['foot_x'] for r in leg2_results]):.4f}")
print(f"  Z range: {min([r['foot_z'] for r in leg2_results]):.4f} to {max([r['foot_z'] for r in leg2_results]):.4f}")

print("\n" + "="*60)
print("\n=== RECOMMENDATION ===\n")

print("For a walking gait, knees should:")
print("1. Keep feet on ground during stance phase (z ≈ 0)")
print("2. Allow feet to swing up for clearance (z > 0)")
print("3. Move smoothly between positions\n")

print("SUGGESTED LIMITS based on analysis:")
if ground_touching_leg1:
    min_ang = min([r['degrees'] for r in ground_touching_leg1])
    max_ang = max([r['degrees'] for r in ground_touching_leg1])
    print(f"  Knee 1: {min_ang:.1f}° to {max_ang:.1f}°")
    print(f"           or {np.radians(min_ang):.3f} to {np.radians(max_ang):.3f} radians")

if ground_touching_leg2:
    min_ang = min([r['degrees'] for r in ground_touching_leg2])
    max_ang = max([r['degrees'] for r in ground_touching_leg2])
    print(f"  Knee 2: {min_ang:.1f}° to {max_ang:.1f}°")
    print(f"           or {np.radians(min_ang):.3f} to {np.radians(max_ang):.3f} radians")
