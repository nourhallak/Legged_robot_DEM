"""
Check robot geometry - foot offset and body orientation
"""

import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

# Print all body info
print("="*70)
print("ROBOT GEOMETRY - BODY POSITIONS")
print("="*70)

body_names = ['body', 'hip', 'thigh_1', 'leg_1', 'foot_1', 'thigh_2', 'leg_2', 'foot_2']

for name in body_names:
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        pos = model.body_pos[body_id]
        quat = model.body_quat[body_id]
        print(f"\n{name}:")
        print(f"  Position: X={pos[0]:8.4f}, Y={pos[1]:8.4f}, Z={pos[2]:8.4f} (m)")
        print(f"  Rotation: {quat}")
    except:
        pass

# Check at zero angles
print("\n" + "="*70)
print("FOOT POSITIONS AT ZERO JOINT ANGLES")
print("="*70)

mujoco.mj_forward(model, data)

foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'foot_1')
foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'foot_2')

foot1_pos = data.xpos[foot1_id]
foot2_pos = data.xpos[foot2_id]

print(f"\nFoot 1 (Left):")
print(f"  X: {foot1_pos[0]*1000:7.1f}mm")
print(f"  Y: {foot1_pos[1]*1000:7.1f}mm")
print(f"  Z: {foot1_pos[2]*1000:7.1f}mm")

print(f"\nFoot 2 (Right):")
print(f"  X: {foot2_pos[0]*1000:7.1f}mm")
print(f"  Y: {foot2_pos[1]*1000:7.1f}mm")
print(f"  Z: {foot2_pos[2]*1000:7.1f}mm")

print(f"\nLateral separation: {abs(foot1_pos[1] - foot2_pos[1])*1000:.1f}mm")

# Check trajectory expected positions
import numpy as np
from pathlib import Path

traj_file = Path(__file__).parent / "biped_10step_sinusoidal.npy"
traj_data = np.load(str(traj_file), allow_pickle=True).item()
left_trajectory = traj_data['left_trajectory']
right_trajectory = traj_data['right_trajectory']

print("\n" + "="*70)
print("PLANNED TRAJECTORY - INITIAL POSITION")
print("="*70)

print(f"\nLeft foot (frame 0):")
print(f"  X: {left_trajectory[0, 0]*1000:7.1f}mm")
print(f"  Y: {left_trajectory[0, 1]*1000:7.1f}mm")
print(f"  Z: {left_trajectory[0, 2]*1000:7.1f}mm")

print(f"\nRight foot (frame 0):")
print(f"  X: {right_trajectory[0, 0]*1000:7.1f}mm")
print(f"  Y: {right_trajectory[0, 1]*1000:7.1f}mm")
print(f"  Z: {right_trajectory[0, 2]*1000:7.1f}mm")

print(f"\nPlanned lateral separation: {abs(left_trajectory[0, 1] - right_trajectory[0, 1])*1000:.1f}mm")

print("\n" + "="*70)
print("MISMATCH ANALYSIS")
print("="*70)
print(f"\nActual foot Y positions: Left={foot1_pos[1]*1000:.1f}, Right={foot2_pos[1]*1000:.1f}")
print(f"Planned foot Y positions: Left={left_trajectory[0, 1]*1000:.1f}, Right={right_trajectory[0, 1]*1000:.1f}")
print(f"\nLeft Y offset: {(foot1_pos[1] - left_trajectory[0, 1])*1000:.1f}mm")
print(f"Right Y offset: {(foot2_pos[1] - right_trajectory[0, 1])*1000:.1f}mm")
