"""
Detailed trajectory error breakdown - identify which coordinates are problematic
"""

import numpy as np
import mujoco
from pathlib import Path

# Load data
model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

traj_file = Path(__file__).parent / "biped_10step_sinusoidal.npy"
traj_data = np.load(str(traj_file), allow_pickle=True).item()
left_trajectory_planned = traj_data['left_trajectory']
right_trajectory_planned = traj_data['right_trajectory']
times = traj_data['times']

ik_file = Path(__file__).parent / "biped_ik_solutions.npy"
ik_data = np.load(str(ik_file), allow_pickle=True).item()
left_angles = ik_data['left_joint_angles']
right_angles = ik_data['right_joint_angles']

foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'foot_1')
foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'foot_2')

# Simulate and compare
left_trajectory_actual = []
right_trajectory_actual = []

for i in range(len(times)):
    data.qpos[3] = left_angles[i, 0]
    data.qpos[4] = left_angles[i, 1]
    data.qpos[5] = left_angles[i, 2]
    data.qpos[6] = right_angles[i, 0]
    data.qpos[7] = right_angles[i, 1]
    data.qpos[8] = right_angles[i, 2]
    
    mujoco.mj_forward(model, data)
    
    foot1_pos = data.xpos[foot1_id]
    foot2_pos = data.xpos[foot2_id]
    
    left_trajectory_actual.append([foot1_pos[0], foot1_pos[1], foot1_pos[2]])
    right_trajectory_actual.append([foot2_pos[0], foot2_pos[1], foot2_pos[2]])

left_trajectory_actual = np.array(left_trajectory_actual)
right_trajectory_actual = np.array(right_trajectory_actual)

# Analyze by coordinate
print("="*70)
print("ERROR BREAKDOWN BY COORDINATE")
print("="*70)

coords = ['X (Forward)', 'Y (Lateral)', 'Z (Height)']

for leg_name, planned, actual in [('LEFT LEG', left_trajectory_planned, left_trajectory_actual),
                                    ('RIGHT LEG', right_trajectory_planned, right_trajectory_actual)]:
    print(f"\n{leg_name}:")
    total_error = np.linalg.norm(planned - actual, axis=1)
    
    for coord_idx, coord_name in enumerate(coords):
        coord_error = np.abs(planned[:, coord_idx] - actual[:, coord_idx])
        print(f"  {coord_name}:")
        print(f"    Mean: {coord_error.mean()*1000:7.2f}mm  |  Max: {coord_error.max()*1000:7.2f}mm  |  Std: {coord_error.std()*1000:7.2f}mm")

# Show specific problem frames
print("\n" + "="*70)
print("WORST ERROR FRAMES")
print("="*70)

left_total_error = np.linalg.norm(left_trajectory_planned - left_trajectory_actual, axis=1)
right_total_error = np.linalg.norm(right_trajectory_planned - right_trajectory_actual, axis=1)

left_worst_idx = np.argsort(left_total_error)[-5:][::-1]
right_worst_idx = np.argsort(right_total_error)[-5:][::-1]

print("\nLEFT LEG WORST 5:")
for idx in left_worst_idx:
    err = left_total_error[idx]*1000
    planned = left_trajectory_planned[idx]
    actual = left_trajectory_actual[idx]
    print(f"  Frame {idx} ({times[idx]:.2f}s): Error={err:.2f}mm")
    print(f"    Planned: X={planned[0]*1000:.1f}, Y={planned[1]*1000:.1f}, Z={planned[2]*1000:.1f}")
    print(f"    Actual:  X={actual[0]*1000:.1f}, Y={actual[1]*1000:.1f}, Z={actual[2]*1000:.1f}")
    dx = (actual[0] - planned[0])*1000
    dy = (actual[1] - planned[1])*1000
    dz = (actual[2] - planned[2])*1000
    print(f"    Delta:   dX={dx:+.1f}, dY={dy:+.1f}, dZ={dz:+.1f}")

print("\nRIGHT LEG WORST 5:")
for idx in right_worst_idx:
    err = right_total_error[idx]*1000
    planned = right_trajectory_planned[idx]
    actual = right_trajectory_actual[idx]
    print(f"  Frame {idx} ({times[idx]:.2f}s): Error={err:.2f}mm")
    print(f"    Planned: X={planned[0]*1000:.1f}, Y={planned[1]*1000:.1f}, Z={planned[2]*1000:.1f}")
    print(f"    Actual:  X={actual[0]*1000:.1f}, Y={actual[1]*1000:.1f}, Z={actual[2]*1000:.1f}")
    dx = (actual[0] - planned[0])*1000
    dy = (actual[1] - planned[1])*1000
    dz = (actual[2] - planned[2])*1000
    print(f"    Delta:   dX={dx:+.1f}, dY={dy:+.1f}, dZ={dz:+.1f}")
