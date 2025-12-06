"""
Check foot contact during stance phase
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

print("="*70)
print("FOOT HEIGHT ANALYSIS DURING WALKING")
print("="*70)

# Check stance phases (when foot is on ground)
# Based on trajectory: frames 0-50 = left swing, 50-100 = left stance
# 100-150 = right swing, 150-200 = right stance, etc.

# Frame 75 = middle of left stance
idx_left_stance = 75
data.qpos[3] = left_angles[idx_left_stance, 0]
data.qpos[4] = left_angles[idx_left_stance, 1]
data.qpos[5] = left_angles[idx_left_stance, 2]
data.qpos[6] = right_angles[idx_left_stance, 0]
data.qpos[7] = right_angles[idx_left_stance, 1]
data.qpos[8] = right_angles[idx_left_stance, 2]
mujoco.mj_forward(model, data)

foot1_pos = data.xpos[foot1_id]
foot2_pos = data.xpos[foot2_id]

print(f"\nDuring LEFT LEG STANCE (frame {idx_left_stance}):")
print(f"  Left foot (stance):  Z = {foot1_pos[2]*1000:.2f}mm")
print(f"  Right foot (swing):  Z = {foot2_pos[2]*1000:.2f}mm")
print(f"  Floor Z:             {0.431*1000:.2f}mm")
print(f"  Left foot gap from floor:  {(foot1_pos[2] - 0.431)*1000:+.2f}mm")
print(f"  Right foot gap from floor: {(foot2_pos[2] - 0.431)*1000:+.2f}mm")

# Frame 175 = middle of right stance
idx_right_stance = 175
data.qpos[3] = left_angles[idx_right_stance, 0]
data.qpos[4] = left_angles[idx_right_stance, 1]
data.qpos[5] = left_angles[idx_right_stance, 2]
data.qpos[6] = right_angles[idx_right_stance, 0]
data.qpos[7] = right_angles[idx_right_stance, 1]
data.qpos[8] = right_angles[idx_right_stance, 2]
mujoco.mj_forward(model, data)

foot1_pos = data.xpos[foot1_id]
foot2_pos = data.xpos[foot2_id]

print(f"\nDuring RIGHT LEG STANCE (frame {idx_right_stance}):")
print(f"  Left foot (swing):   Z = {foot1_pos[2]*1000:.2f}mm")
print(f"  Right foot (stance): Z = {foot2_pos[2]*1000:.2f}mm")
print(f"  Floor Z:             {0.431*1000:.2f}mm")
print(f"  Left foot gap from floor:  {(foot1_pos[2] - 0.431)*1000:+.2f}mm")
print(f"  Right foot gap from floor: {(foot2_pos[2] - 0.431)*1000:+.2f}mm")

# Check at zero angles
data.qpos[3:9] = 0
mujoco.mj_forward(model, data)
foot1_pos = data.xpos[foot1_id]
foot2_pos = data.xpos[foot2_id]

print(f"\nAt ZERO ANGLES (standing):")
print(f"  Left foot:  Z = {foot1_pos[2]*1000:.2f}mm")
print(f"  Right foot: Z = {foot2_pos[2]*1000:.2f}mm")
print(f"  Floor Z:    {0.431*1000:.2f}mm")
print(f"  Both feet gap: {(foot1_pos[2] - 0.431)*1000:+.2f}mm")

print("\n" + "="*70)
if (foot1_pos[2] - 0.431) < 0.001:
    print("✅ Feet are touching ground at standing position")
else:
    print(f"❌ Feet are {(foot1_pos[2] - 0.431)*1000:.2f}mm ABOVE ground")
    print(f"   Need to RAISE floor or LOWER standing height")
