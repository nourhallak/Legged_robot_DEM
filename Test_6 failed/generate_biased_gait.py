"""
Generate a forward-BIASED walking gait
Left leg pushes BACK (propels forward), right leg swings FORWARD
"""
import numpy as np
from scipy.interpolate import interp1d

gait_period = 3.0
half_period = gait_period / 2.0
t = np.linspace(0, gait_period, 100)

# ===== LEFT LEG (Push/Stance phase) =====
l_hip = np.zeros_like(t)
l_knee = np.zeros_like(t)
l_ankle = np.zeros_like(t)

for i, time in enumerate(t):
    if time < half_period:
        # LEFT LEG: STANCE - Push hard backward
        alpha = time / half_period
        l_hip[i] = -0.3 * np.sin(np.pi * alpha)  # Rotate hip backward
        l_knee[i] = -0.8 - 0.2 * np.sin(np.pi * alpha)  # Bent, compressing
        l_ankle[i] = -0.2  # Point down
    else:
        # LEFT LEG: SWING - Bring forward quickly
        alpha = (time - half_period) / half_period
        l_hip[i] = 0.2 * np.sin(np.pi * alpha)  # Rotate forward
        l_knee[i] = -0.4 + 0.3 * np.cos(np.pi * alpha)  # Bent for clearance
        l_ankle[i] = 0.3 + 0.1 * np.cos(np.pi * alpha)  # Dorsiflexed

# ===== RIGHT LEG (Swing phase, opposite) =====
r_hip = np.zeros_like(t)
r_knee = np.zeros_like(t)
r_ankle = np.zeros_like(t)

for i, time in enumerate(t):
    if time < half_period:
        # RIGHT LEG: SWING
        alpha = time / half_period
        r_hip[i] = 0.2 * np.sin(np.pi * alpha)  # Rotate forward
        r_knee[i] = -0.4 + 0.3 * np.cos(np.pi * alpha)  # Bent for clearance
        r_ankle[i] = 0.3 + 0.1 * np.cos(np.pi * alpha)  # Dorsiflexed
    else:
        # RIGHT LEG: STANCE - Push hard backward
        alpha = (time - half_period) / half_period
        r_hip[i] = -0.3 * np.sin(np.pi * alpha)  # Rotate hip backward
        r_knee[i] = -0.8 - 0.2 * np.sin(np.pi * alpha)  # Bent, compressing
        r_ankle[i] = -0.2  # Point down

# Save
np.save('ik_times.npy', t)
np.save('ik_left_hip.npy', l_hip)
np.save('ik_left_knee.npy', l_knee)
np.save('ik_left_ankle.npy', l_ankle)
np.save('ik_right_hip.npy', r_hip)
np.save('ik_right_knee.npy', r_knee)
np.save('ik_right_ankle.npy', r_ankle)

print("=" * 80)
print("FORWARD-BIASED WALKING GAIT")
print("=" * 80)
print(f"Gait period: {gait_period}s\n")

print("LEFT LEG (Stance 0-1.5s, Swing 1.5-3.0s)")
print(f"  Hip:   {l_hip.min():.4f} to {l_hip.max():.4f} (backward push to forward swing)")
print(f"  Knee:  {l_knee.min():.4f} to {l_knee.max():.4f}")
print(f"  Ankle: {l_ankle.min():.4f} to {l_ankle.max():.4f}")

print("\nRIGHT LEG (Swing 0-1.5s, Stance 1.5-3.0s)")
print(f"  Hip:   {r_hip.min():.4f} to {r_hip.max():.4f}")
print(f"  Knee:  {r_knee.min():.4f} to {r_knee.max():.4f}")
print(f"  Ankle: {r_ankle.min():.4f} to {r_ankle.max():.4f}")

print("\n[+] Trajectories saved!")
print("[+] This gait creates:   LEFT_PUSH → RIGHT_PUSH → LEFT_PUSH (continuous forward motion)")
