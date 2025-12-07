"""
Debug: Check what the forward gait trajectories are actually doing
"""
import numpy as np
from scipy.interpolate import interp1d

# Load trajectories
ik_times = np.load('ik_times.npy')
l_hip = np.load('ik_left_hip.npy') * 0.20
r_hip = np.load('ik_right_hip.npy') * 0.20

print("Left Hip Motion (forward stepping):")
print(f"  Min: {l_hip.min():.4f} (backward)")
print(f"  Max: {l_hip.max():.4f} (forward)")
print(f"  Amplitude: {l_hip.max() - l_hip.min():.4f}")

print("\nRight Hip Motion:")
print(f"  Min: {r_hip.min():.4f}")
print(f"  Max: {r_hip.max():.4f}")
print(f"  Amplitude: {r_hip.max() - r_hip.min():.4f}")

print("\nSample values:")
for i in [0, 25, 50, 75, 99]:
    print(f"  t={ik_times[i]:.2f}: L_hip={l_hip[i]:+.4f}, R_hip={r_hip[i]:+.4f}")
