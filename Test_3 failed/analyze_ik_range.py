import numpy as np
import mujoco

# Load model and trajectories
model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

traj_data = np.load('biped_10step_sinusoidal.npy', allow_pickle=True).item()
ik_data = np.load('biped_ik_solutions.npy', allow_pickle=True).item()

left_traj = traj_data['left_trajectory']
right_traj = traj_data['right_trajectory']
left_ik = ik_data['left_joint_angles']
right_ik = ik_data['right_joint_angles']

print("Checking foot positions and IK joint angles:\n")
print("EARLY STEPS (should be normal):")
for i in [50, 100, 150]:
    print(f"\nFrame {i}:")
    print(f"  Left foot target:  X={left_traj[i,0]*1000:7.2f}mm | Z={left_traj[i,2]*1000:7.2f}mm")
    print(f"  Left IK angles:    [{left_ik[i,0]:.3f}, {left_ik[i,1]:.3f}, {left_ik[i,2]:.3f}]")
    print(f"  Right foot target: X={right_traj[i,0]*1000:7.2f}mm | Z={right_traj[i,2]*1000:7.2f}mm")
    print(f"  Right IK angles:   [{right_ik[i,0]:.3f}, {right_ik[i,1]:.3f}, {right_ik[i,2]:.3f}]")

print("\n\nLATE STEPS (might be problematic):")
for i in [400, 450, 490]:
    print(f"\nFrame {i}:")
    print(f"  Left foot target:  X={left_traj[i,0]*1000:7.2f}mm | Z={left_traj[i,2]*1000:7.2f}mm")
    print(f"  Left IK angles:    [{left_ik[i,0]:.3f}, {left_ik[i,1]:.3f}, {left_ik[i,2]:.3f}]")
    print(f"  Right foot target: X={right_traj[i,0]*1000:7.2f}mm | Z={right_traj[i,2]*1000:7.2f}mm")
    print(f"  Right IK angles:   [{right_ik[i,0]:.3f}, {right_ik[i,1]:.3f}, {right_ik[i,2]:.3f}]")

print("\n\nMax joint angle ranges (from model):")
print("Hip joint:    [-1.57, 1.57]")
print("Knee joint:   [-2.09, 1.05]")
print("Ankle joint:  [-1.57, 1.57]")

print(f"\n\nLeft leg angle statistics:")
print(f"  Hip:   min={np.min(left_ik[:,0]):.3f}, max={np.max(left_ik[:,0]):.3f}")
print(f"  Knee:  min={np.min(left_ik[:,1]):.3f}, max={np.max(left_ik[:,1]):.3f}")
print(f"  Ankle: min={np.min(left_ik[:,2]):.3f}, max={np.max(left_ik[:,2]):.3f}")

print(f"\nRight leg angle statistics:")
print(f"  Hip:   min={np.min(right_ik[:,0]):.3f}, max={np.max(right_ik[:,0]):.3f}")
print(f"  Knee:  min={np.min(right_ik[:,1]):.3f}, max={np.max(right_ik[:,1]):.3f}")
print(f"  Ankle: min={np.min(right_ik[:,2]):.3f}, max={np.max(right_ik[:,2]):.3f}")

# Find problematic frames
print("\n\nFrames with extreme joint angles (>1.5 rad):")
for i in range(len(left_ik)):
    if np.any(np.abs(left_ik[i]) > 1.5) or np.any(np.abs(right_ik[i]) > 1.5):
        print(f"  Frame {i}: Left={left_ik[i]} | Right={right_ik[i]}")
