"""
Analyze foot contact throughout the entire walking sequence
"""
import numpy as np
import mujoco
from pathlib import Path

model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

# Load trajectories and IK
traj_file = Path(__file__).parent / "biped_10step_sinusoidal.npy"
traj_data = np.load(str(traj_file), allow_pickle=True).item()
times = traj_data['times']

ik_file = Path(__file__).parent / "biped_ik_solutions.npy"
ik_data = np.load(str(ik_file), allow_pickle=True).item()
left_angles = ik_data['left_joint_angles']
right_angles = ik_data['right_joint_angles']

foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'foot_1')
foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'foot_2')

print("="*80)
print("FOOT HEIGHT THROUGHOUT WALKING SEQUENCE")
print("="*80)

# Each step is 100 frames (50 swing + 50 stance)
# Check middle of stance for each step
floor_z = 0.431

for step in range(1, 11):
    # Alternate between left and right
    if step % 2 == 1:  # Odd steps = left swings
        # Left stance at end of step
        frame_idx = (step * 100) - 25  # Middle of stance
        stance_leg = "LEFT"
    else:  # Even steps = right swings
        # Right stance at end of step
        frame_idx = (step * 100) - 25  # Middle of stance
        stance_leg = "RIGHT"
    
    if frame_idx >= len(times):
        break
    
    # Set joint angles
    data.qpos[3] = left_angles[frame_idx, 0]
    data.qpos[4] = left_angles[frame_idx, 1]
    data.qpos[5] = left_angles[frame_idx, 2]
    data.qpos[6] = right_angles[frame_idx, 0]
    data.qpos[7] = right_angles[frame_idx, 1]
    data.qpos[8] = right_angles[frame_idx, 2]
    mujoco.mj_forward(model, data)
    
    foot1_pos = data.xpos[foot1_id]
    foot2_pos = data.xpos[foot2_id]
    
    left_gap = (foot1_pos[2] - floor_z) * 1000
    right_gap = (foot2_pos[2] - floor_z) * 1000
    
    print(f"\nStep {step:2d} ({stance_leg} stance)  Frame {frame_idx:4d}")
    print(f"  Left foot:  Z={foot1_pos[2]*1000:.2f}mm  Gap={left_gap:+6.2f}mm", end="")
    if left_gap > 1.5:
        print("  ⚠️  FLOATING")
    else:
        print("  ✓ Contact")
    
    print(f"  Right foot: Z={foot2_pos[2]*1000:.2f}mm  Gap={right_gap:+6.2f}mm", end="")
    if right_gap > 1.5:
        print("  ⚠️  FLOATING")
    else:
        print("  ✓ Contact")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Check if there's a pattern in the IK errors that grows
print("\nChecking IK errors across steps...")
for step in range(1, 11):
    frame_idx = (step * 100) - 25
    if frame_idx >= len(left_angles):
        break
    
    # Get the IK solution
    data.qpos[3] = left_angles[frame_idx, 0]
    data.qpos[4] = left_angles[frame_idx, 1]
    data.qpos[5] = left_angles[frame_idx, 2]
    data.qpos[6] = right_angles[frame_idx, 0]
    data.qpos[7] = right_angles[frame_idx, 1]
    data.qpos[8] = right_angles[frame_idx, 2]
    mujoco.mj_forward(model, data)
    
    # Get trajectory target
    traj_target = traj_data['left_trajectory'][frame_idx]
    actual = data.xpos[foot1_id]
    
    error = np.linalg.norm(traj_target - actual[:3])
    print(f"Step {step}: Left leg IK error = {error*1000:.2f}mm")
