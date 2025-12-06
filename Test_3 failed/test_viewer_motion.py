"""
Test if the viewer shows motion with new IK solutions
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import time

# Load model
model_path = Path(__file__).parent / "legged_robot_ik.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Load IK solutions
ik_file = Path(__file__).parent / "biped_ik_solutions.npy"
ik_data = np.load(str(ik_file), allow_pickle=True).item()
left_angles = ik_data['left_joint_angles']
right_angles = ik_data['right_joint_angles']

print(f"Loaded {len(left_angles)} joint angle sets")
print(f"Left leg first frame: hip={np.degrees(left_angles[0,0]):.2f}°, knee={np.degrees(left_angles[0,1]):.2f}°, ankle={np.degrees(left_angles[0,2]):.2f}°")
print(f"Right leg first frame: hip={np.degrees(right_angles[0,0]):.2f}°, knee={np.degrees(right_angles[0,1]):.2f}°, ankle={np.degrees(right_angles[0,2]):.2f}°")

# Test setting joint angles
print("\nTesting joint angle application...")

# Set to frame 0
data.qpos[3] = left_angles[0, 0]   # Left hip
data.qpos[4] = left_angles[0, 1]   # Left knee  
data.qpos[5] = left_angles[0, 2]   # Left ankle
data.qpos[6] = right_angles[0, 0]  # Right hip
data.qpos[7] = right_angles[0, 1]  # Right knee
data.qpos[8] = right_angles[0, 2]  # Right ankle
mujoco.mj_forward(model, data)

print(f"Frame 0: qpos set successfully")

# Set to frame 250 (middle of motion)
data.qpos[3] = left_angles[250, 0]
data.qpos[4] = left_angles[250, 1]
data.qpos[5] = left_angles[250, 2]
data.qpos[6] = right_angles[250, 0]
data.qpos[7] = right_angles[250, 1]
data.qpos[8] = right_angles[250, 2]
mujoco.mj_forward(model, data)

print(f"Frame 250: qpos set successfully")
print(f"  Left leg: hip={np.degrees(data.qpos[3]):.2f}°, knee={np.degrees(data.qpos[4]):.2f}°")

print("\nLaunching viewer to test motion...")
print("You should see the robot cycle through walking motions")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -25
    
    frame_idx = 0
    frame_skip = 2  # Show every 2nd frame
    
    while viewer.is_running() and frame_idx < len(left_angles):
        with viewer.lock():
            # Set joint angles directly
            data.qpos[3] = left_angles[frame_idx, 0]
            data.qpos[4] = left_angles[frame_idx, 1]
            data.qpos[5] = left_angles[frame_idx, 2]
            data.qpos[6] = right_angles[frame_idx, 0]
            data.qpos[7] = right_angles[frame_idx, 1]
            data.qpos[8] = right_angles[frame_idx, 2]
            
            mujoco.mj_forward(model, data)
            
            frame_idx += frame_skip
            
            if frame_idx % 100 == 0:
                print(f"Frame {frame_idx}/{len(left_angles)}")
        
        # Slow down display
        time.sleep(0.01)

print("Done!")
