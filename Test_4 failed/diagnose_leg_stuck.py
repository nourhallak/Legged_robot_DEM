#!/usr/bin/env python3
"""
Diagnose which leg is stuck during simulation
"""

import numpy as np
import mujoco
import re
from pathlib import Path

def load_model_with_assets(xml_path="legged_robot_ik.xml"):
    xml_file = Path(xml_path)
    if not xml_file.exists():
        xml_file = Path(".") / "legged_robot_ik.xml"
    
    with open(xml_file, "r") as f:
        xml_content = f.read()
    
    asset_dir = xml_file.parent / "Legged_robot" / "meshes"
    
    for match in re.finditer(r'filename="([^"]*\.stl)"', xml_content):
        stl_file = match.group(1)
        full_path = asset_dir / stl_file
        
        if full_path.exists():
            xml_content = xml_content.replace(f'filename="{stl_file}"', f'filename="{full_path}"')
    
    return mujoco.MjModel.from_xml_string(xml_content)

print("="*80)
print("LEG MOTION DIAGNOSIS")
print("="*80)

# Load model
model = load_model_with_assets()
data = mujoco.MjData(model)

# Load trajectories and solutions
base_traj = np.load("base_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")
qpos_solutions = np.load("joint_solutions_ik.npy")

# Initialize
data.qpos[:] = qpos_solutions[0]
mujoco.mj_forward(model, data)

# Get body/site IDs
foot1_site_id = model.site(name='foot1_site').id
foot2_site_id = model.site(name='foot2_site').id
foot1_body_id = model.body(name='foot_1').id
foot2_body_id = model.body(name='foot_2').id

print("\nControl parameters:")
Kp = 500.0
Kd = 50.0
frames_per_step = 100
print(f"  Kp={Kp}, Kd={Kd}")
print(f"  Frames per trajectory step: {frames_per_step}")

# Simulate and collect data
print("\nSimulating and collecting motion data...")

leg1_errors = []
leg2_errors = []
leg1_positions = []
leg2_positions = []

step_idx = 0
frame_count = 0
num_steps = len(qpos_solutions)

for total_frame in range(frames_per_step * 20):  # Simulate 20 trajectory steps
    # Get target
    qpos_target = qpos_solutions[step_idx % num_steps]
    
    # Apply PD control
    act_joints = [3, 4, 5, 6, 7, 8]
    for i, j in enumerate(act_joints):
        error = qpos_target[j] - data.qpos[j]
        vel = data.qvel[j]
        ctrl = Kp * error - Kd * vel
        data.ctrl[i] = np.clip(ctrl, -10.0, 10.0)
    
    # Step simulation
    mujoco.mj_step(model, data)
    frame_count += 1
    
    # Advance trajectory
    if frame_count >= frames_per_step:
        frame_count = 0
        step_idx += 1
        
        # Record motion at trajectory transitions
        if step_idx < 20:
            data.qpos[:] = data.qpos  # Current state
            mujoco.mj_forward(model, data)
            
            # Get foot positions
            foot1_pos = data.site_xpos[foot1_site_id].copy()
            foot2_pos = data.site_xpos[foot2_site_id].copy()
            
            foot1_target = foot1_traj[step_idx - 1]
            foot2_target = foot2_traj[step_idx - 1]
            
            err1 = np.linalg.norm(foot1_pos - foot1_target)
            err2 = np.linalg.norm(foot2_pos - foot2_target)
            
            leg1_errors.append(err1)
            leg2_errors.append(err2)
            leg1_positions.append(foot1_pos.copy())
            leg2_positions.append(foot2_pos.copy())
            
            # Get joint angles
            leg1_q = data.qpos[[3, 4, 5]]
            leg2_q = data.qpos[[6, 7, 8]]
            
            print(f"\nStep {step_idx}:")
            print(f"  Leg1 (Left):")
            print(f"    Joint angles: [{np.degrees(leg1_q[0]):.1f}°, {np.degrees(leg1_q[1]):.1f}°, {np.degrees(leg1_q[2]):.1f}°]")
            print(f"    Position: {foot1_pos}")
            print(f"    Target:   {foot1_target}")
            print(f"    Error: {err1*1000:.2f}mm")
            
            print(f"  Leg2 (Right):")
            print(f"    Joint angles: [{np.degrees(leg2_q[0]):.1f}°, {np.degrees(leg2_q[1]):.1f}°, {np.degrees(leg2_q[2]):.1f}°]")
            print(f"    Position: {foot2_pos}")
            print(f"    Target:   {foot2_target}")
            print(f"    Error: {err2*1000:.2f}mm")

print("\n" + "="*80)
print("MOTION SUMMARY")
print("="*80)

leg1_errors = np.array(leg1_errors)
leg2_errors = np.array(leg2_errors)

print(f"\nLeg1 (Left):")
print(f"  Mean error: {leg1_errors.mean()*1000:.2f}mm")
print(f"  Max error:  {leg1_errors.max()*1000:.2f}mm")
print(f"  Min error:  {leg1_errors.min()*1000:.2f}mm")

print(f"\nLeg2 (Right):")
print(f"  Mean error: {leg2_errors.mean()*1000:.2f}mm")
print(f"  Max error:  {leg2_errors.max()*1000:.2f}mm")
print(f"  Min error:  {leg2_errors.min()*1000:.2f}mm")

# Check if one leg is stuck
leg1_motion = np.array(leg1_positions)
leg2_motion = np.array(leg2_positions)

leg1_displacement = np.linalg.norm(leg1_motion[1:] - leg1_motion[:-1], axis=1)
leg2_displacement = np.linalg.norm(leg2_motion[1:] - leg2_motion[:-1], axis=1)

print(f"\nLeg displacement per step:")
print(f"  Leg1: mean={leg1_displacement.mean()*1000:.2f}mm, max={leg1_displacement.max()*1000:.2f}mm")
print(f"  Leg2: mean={leg2_displacement.mean()*1000:.2f}mm, max={leg2_displacement.max()*1000:.2f}mm")

if leg1_displacement.mean() < 0.5e-3 and leg2_displacement.mean() > 1e-3:
    print("\n[DIAGNOSIS] Leg1 (Left) is STUCK - minimal motion")
elif leg2_displacement.mean() < 0.5e-3 and leg1_displacement.mean() > 1e-3:
    print("\n[DIAGNOSIS] Leg2 (Right) is STUCK - minimal motion")
else:
    print("\n[DIAGNOSIS] Both legs moving properly")
