import numpy as np

# Load trajectories
traj_data = np.load('biped_10step_sinusoidal.npy', allow_pickle=True).item()

left_traj = traj_data['left_trajectory']
right_traj = traj_data['right_trajectory']

print("LEFT TRAJECTORY:")
for i in [37, 87, 137, 187, 237, 287, 337, 387, 437, 487]:
    print(f"  Frame {i:3d}: Z={left_traj[i, 2]*1000:7.2f}mm | X={left_traj[i, 0]*1000:7.2f}mm")

print("\nRIGHT TRAJECTORY:")
for i in [37, 87, 137, 187, 237, 287, 337, 387, 437, 487]:
    print(f"  Frame {i:3d}: Z={right_traj[i, 2]*1000:7.2f}mm | X={right_traj[i, 0]*1000:7.2f}mm")

print("\n\nStanding height:", 0.431, "m =", 431, "mm")
print("Step height:", traj_data['step_height']*1000, "mm")

# Check for swing phase peaks
print("\nLEFT SWING PEAKS:")
for step in range(0, 500, 50):
    step_end = min(step + 50, 500)
    z_max = np.max(left_traj[step:step_end, 2])
    z_min = np.min(left_traj[step:step_end, 2])
    print(f"  Step {step:3d}-{step_end:3d}: Z_max={z_max*1000:.2f}mm, Z_min={z_min*1000:.2f}mm")

print("\nRIGHT SWING PEAKS:")
for step in range(0, 500, 50):
    step_end = min(step + 50, 500)
    z_max = np.max(right_traj[step:step_end, 2])
    z_min = np.min(right_traj[step:step_end, 2])
    print(f"  Step {step:3d}-{step_end:3d}: Z_max={z_max*1000:.2f}mm, Z_min={z_min*1000:.2f}mm")
