"""
Generate a PROPER forward walking gait
Focus on knee extension/compression for pushing against sand
AND hip rotation for step timing
"""
import numpy as np
from scipy.interpolate import interp1d

# Gait cycle
gait_period = 3.0
half_period = gait_period / 2.0
t = np.linspace(0, gait_period, 100)

# ===== LEFT LEG =====
# Stance: 0 to 1.5s - push into sand
# Swing: 1.5 to 3.0s - lift and swing forward

# L-Hip: Rotate to step (small angles OK, just for timing)
l_hip = np.zeros_like(t)
for i, time in enumerate(t):
    if time < half_period:
        # Stance: rotate hip backward slightly
        l_hip[i] = -0.1 * np.sin(np.pi * time / half_period)
    else:
        # Swing: rotate hip forward
        alpha = (time - half_period) / half_period
        l_hip[i] = 0.1 * np.sin(np.pi * alpha)

# L-Knee: KEY - compress in stance, extend in swing
l_knee = np.zeros_like(t)
for i, time in enumerate(t):
    if time < half_period:
        # Stance: knee COMPRESSED (negative angle = bent)
        # Start bent, compress more, then extend as we push
        alpha = time / half_period
        l_knee[i] = -0.5 - 0.3 * np.sin(np.pi * alpha)**2
    else:
        # Swing: knee EXTENDED (positive/less negative) for clearance
        alpha = (time - half_period) / half_period
        # Bent for clearance during swing
        l_knee[i] = -0.4 + 0.2 * np.cos(np.pi * alpha)

# L-Ankle: Point down in stance (pushing), dorsiflexed in swing
l_ankle = np.zeros_like(t)
for i, time in enumerate(t):
    if time < half_period:
        # Stance: ankle points down
        l_ankle[i] = -0.1
    else:
        # Swing: ankle dorsiflexed (positive = up)
        alpha = (time - half_period) / half_period
        l_ankle[i] = 0.2 + 0.1 * np.cos(np.pi * alpha)

# ===== RIGHT LEG =====
# Opposite phase

# R-Hip
r_hip = np.zeros_like(t)
for i, time in enumerate(t):
    if time < half_period:
        # Swing phase
        r_hip[i] = 0.1 * np.sin(np.pi * time / half_period)
    else:
        # Stance phase
        alpha = (time - half_period) / half_period
        r_hip[i] = -0.1 * np.sin(np.pi * alpha)

# R-Knee
r_knee = np.zeros_like(t)
for i, time in enumerate(t):
    if time < half_period:
        # Swing phase
        alpha = time / half_period
        r_knee[i] = -0.4 + 0.2 * np.cos(np.pi * alpha)
    else:
        # Stance phase
        alpha = (time - half_period) / half_period
        r_knee[i] = -0.5 - 0.3 * np.sin(np.pi * alpha)**2

# R-Ankle
r_ankle = np.zeros_like(t)
for i, time in enumerate(t):
    if time < half_period:
        # Swing phase
        alpha = time / half_period
        r_ankle[i] = 0.2 + 0.1 * np.cos(np.pi * alpha)
    else:
        # Stance phase
        r_ankle[i] = -0.1

# Save
np.save('ik_times.npy', t)
np.save('ik_left_hip.npy', l_hip)
np.save('ik_left_knee.npy', l_knee)
np.save('ik_left_ankle.npy', l_ankle)
np.save('ik_right_hip.npy', r_hip)
np.save('ik_right_knee.npy', r_knee)
np.save('ik_right_ankle.npy', r_ankle)

print("=" * 80)
print("IMPROVED FORWARD WALKING GAIT - KNEE COMPRESSION FOCUS")
print("=" * 80)
print(f"Gait period: {gait_period}s")
print(f"Points per cycle: {len(t)}\n")

print("LEFT LEG (Stance 0-1.5s, Swing 1.5-3.0s)")
print(f"  Hip rotation: {l_hip.min():.4f} to {l_hip.max():.4f}")
print(f"  Knee angle:   {l_knee.min():.4f} to {l_knee.max():.4f} (more negative = more bent)")
print(f"  Ankle angle:  {l_ankle.min():.4f} to {l_ankle.max():.4f}")

print("\nRIGHT LEG (Swing 0-1.5s, Stance 1.5-3.0s)")
print(f"  Hip rotation: {r_hip.min():.4f} to {r_hip.max():.4f}")
print(f"  Knee angle:   {r_knee.min():.4f} to {r_knee.max():.4f}")
print(f"  Ankle angle:  {r_ankle.min():.4f} to {r_ankle.max():.4f}")

print("\nKey features:")
print("  - Knee compression during stance (pushing)")
print("  - Knee extension during swing (clearance)")
print("  - Ankle points down in stance (pushing into sand)")
print("  - Hip rotation for step timing")
print("[+] Trajectories saved!")
