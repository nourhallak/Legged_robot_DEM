import numpy as np

traj_data = np.load('biped_10step_sinusoidal.npy', allow_pickle=True).item()
left_traj = traj_data['left_trajectory']
right_traj = traj_data['right_trajectory']

print("Trajectory X positions (forward/backward):")
print(f"Frame 0:   Left X={left_traj[0,0]*1000:7.2f}mm, Right X={right_traj[0,0]*1000:7.2f}mm")
print(f"Frame 250: Left X={left_traj[250,0]*1000:7.2f}mm, Right X={right_traj[250,0]*1000:7.2f}mm")
print(f"Frame 499: Left X={left_traj[499,0]*1000:7.2f}mm, Right X={right_traj[499,0]*1000:7.2f}mm")

print("\n\nThe trajectory should LOOP:")
print(f"Start to end forward movement:")
print(f"  Left:  {left_traj[0,0]*1000:.2f}mm -> {left_traj[499,0]*1000:.2f}mm (total: {(left_traj[499,0]-left_traj[0,0])*1000:.2f}mm)")
print(f"  Right: {right_traj[0,0]*1000:.2f}mm -> {right_traj[499,0]*1000:.2f}mm (total: {(right_traj[499,0]-right_traj[0,0])*1000:.2f}mm)")

print("\n\nFor proper looping, frame 499 should match frame 0 position!")
print("Currently it doesn't - the robot walks 15mm per cycle but doesn't reset.")
