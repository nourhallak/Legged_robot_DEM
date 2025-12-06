"""
Check if gait alternation is correct
"""
import numpy as np

traj = np.load('biped_10step_sinusoidal.npy', allow_pickle=True).item()
left_traj = traj['left_trajectory']
right_traj = traj['right_trajectory']

left_z = left_traj[:, 2]
right_z = right_traj[:, 2]

z_min = min(left_z.min(), right_z.min())
z_max = max(left_z.max(), right_z.max())
z_mid = (z_min + z_max) / 2

print(f'Z min: {z_min*1000:.1f}mm, Z max: {z_max*1000:.1f}mm')
print(f'Z mid: {z_mid*1000:.1f}mm\n')
print('Frame | Left Z (mm) | State | Right Z (mm) | State')
print('------|-------------|-------|-------------|-------')

# Check at specific frames
for frame in [0, 50, 100, 150, 200, 250, 300, 350]:
    left_z_mm = left_z[frame]*1000
    right_z_mm = right_z[frame]*1000
    left_state = 'HIGH' if left_z[frame] > z_mid else 'LOW '
    right_state = 'HIGH' if right_z[frame] > z_mid else 'LOW '
    print(f'{frame:5d} | {left_z_mm:11.1f} | {left_state} | {right_z_mm:11.1f} | {right_state}')

# Count how many frames have both legs HIGH
both_high = np.sum((left_z > z_mid) & (right_z > z_mid))
print(f'\nFrames with BOTH legs high: {both_high}/1000')
print(f'This means {both_high/10}% of the gait has flying motion!')
