"""
Generate a REDUCED-amplitude forward-biased walking gait
Keep the robot ON the sand (X: 0.150 to 0.470m = 0.32m total)
"""
import numpy as np

gait_period = 3.0
half_period = gait_period / 2.0
t = np.linspace(0, gait_period, 100)

# REDUCED AMPLITUDES - smaller steps to stay on sand
hip_amp = 0.15  # Reduced from 0.3
knee_amp = 0.15  # Reduced from 0.2
ankle_amp = 0.15  # Reduced from 0.2

# ===== LEFT LEG (Push/Stance phase) =====
l_hip = np.zeros_like(t)
l_knee = np.zeros_like(t)
l_ankle = np.zeros_like(t)

for i, time in enumerate(t):
    if time < half_period:
        # LEFT LEG: STANCE - Push backward
        alpha = time / half_period
        l_hip[i] = -hip_amp * np.sin(np.pi * alpha)
        l_knee[i] = -0.5 - knee_amp * np.sin(np.pi * alpha)
        l_ankle[i] = -0.1
    else:
        # LEFT LEG: SWING - Bring forward
        alpha = (time - half_period) / half_period
        l_hip[i] = hip_amp * 0.6 * np.sin(np.pi * alpha)  # Smaller forward swing
        l_knee[i] = -0.3 + knee_amp * np.cos(np.pi * alpha)
        l_ankle[i] = 0.2 + ankle_amp * np.cos(np.pi * alpha)

# ===== RIGHT LEG (Swing phase, opposite) =====
r_hip = np.zeros_like(t)
r_knee = np.zeros_like(t)
r_ankle = np.zeros_like(t)

for i, time in enumerate(t):
    if time < half_period:
        # RIGHT LEG: SWING
        alpha = time / half_period
        r_hip[i] = hip_amp * 0.6 * np.sin(np.pi * alpha)
        r_knee[i] = -0.3 + knee_amp * np.cos(np.pi * alpha)
        r_ankle[i] = 0.2 + ankle_amp * np.cos(np.pi * alpha)
    else:
        # RIGHT LEG: STANCE - Push backward
        alpha = (time - half_period) / half_period
        r_hip[i] = -hip_amp * np.sin(np.pi * alpha)
        r_knee[i] = -0.5 - knee_amp * np.sin(np.pi * alpha)
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
print("REDUCED-AMPLITUDE FORWARD-BIASED WALKING GAIT")
print("=" * 80)
print(f"Gait period: {gait_period}s")
print(f"Sand grid: X=0.150 to 0.470m (length 0.32m)")
print(f"\nAmplitudes (REDUCED):")
print(f"  Hip swing: ±{hip_amp:.2f} rad")
print(f"  Knee compression: {knee_amp:.2f} rad")
print(f"  Ankle angle: {ankle_amp:.2f} rad")

print(f"\nLEFT LEG (Stance 0-1.5s, Swing 1.5-3.0s)")
print(f"  Hip:   {l_hip.min():.4f} to {l_hip.max():.4f}")
print(f"  Knee:  {l_knee.min():.4f} to {l_knee.max():.4f}")
print(f"  Ankle: {l_ankle.min():.4f} to {l_ankle.max():.4f}")

print(f"\nRIGHT LEG (Swing 0-1.5s, Stance 1.5-3.0s)")
print(f"  Hip:   {r_hip.min():.4f} to {r_hip.max():.4f}")
print(f"  Knee:  {r_knee.min():.4f} to {r_knee.max():.4f}")
print(f"  Ankle: {r_ankle.min():.4f} to {r_ankle.max():.4f}")

print("\n[+] Trajectories saved!")
print("[+] This gait creates smaller steps to keep robot ON sand!")
