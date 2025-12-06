import numpy as np

# Load trajectories
traj_data = np.load('biped_10step_sinusoidal.npy', allow_pickle=True).item()

left_traj = traj_data['left_trajectory']
right_traj = traj_data['right_trajectory']

print("Checking trajectory consistency:")
print(f"Standing height: {0.431*1000}mm")
print(f"Step height: {traj_data['step_height']*1000}mm")

print("\n\nSteps 0-50 (Step 1 - LEFT SWINGS):")
print("  LEFT (swing):")
for i in [0, 12, 25, 37, 49]:
    print(f"    Frame {i:3d}: Z={left_traj[i, 2]*1000:7.2f}mm")
print("  RIGHT (stance):")
for i in [0, 12, 25, 37, 49]:
    print(f"    Frame {i:3d}: Z={right_traj[i, 2]*1000:7.2f}mm")

print("\n\nSteps 50-100 (Step 2 - RIGHT SWINGS):")
print("  LEFT (stance):")
for i in [50, 62, 75, 87, 99]:
    print(f"    Frame {i:3d}: Z={left_traj[i, 2]*1000:7.2f}mm")
print("  RIGHT (swing):")
for i in [50, 62, 75, 87, 99]:
    print(f"    Frame {i:3d}: Z={right_traj[i, 2]*1000:7.2f}mm")
