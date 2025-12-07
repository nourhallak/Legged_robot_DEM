"""
Generate MODERATE-amplitude forward-biased walking gait
Balance: enough forward progress but stays ON sand
"""
import numpy as np

gait_period = 3.0
half_period = gait_period / 2.0
t = np.linspace(0, gait_period, 100)

# MODERATE AMPLITUDES - balance between progress and staying on sand
hip_amp = 0.25  # Between 0.15 and 0.3
knee_amp = 0.18
ankle_amp = 0.15

# ===== LEFT LEG (Push/Stance phase) =====
l_hip = np.zeros_like(t)
l_knee = np.zeros_like(t)
l_ankle = np.zeros_like(t)

for i, time in enumerate(t):
    if time < half_period:
        # LEFT LEG: STANCE - Push backward
        alpha = time / half_period
        l_hip[i] = -hip_amp * np.sin(np.pi * alpha)
        l_knee[i] = -0.6 - knee_amp * np.sin(np.pi * alpha)
        l_ankle[i] = -0.1
    else:
        # LEFT LEG: SWING - Bring forward
        alpha = (time - half_period) / half_period
        l_hip[i] = hip_amp * 0.7 * np.sin(np.pi * alpha)
        l_knee[i] = -0.35 + knee_amp * np.cos(np.pi * alpha)
        l_ankle[i] = 0.2 + ankle_amp * np.cos(np.pi * alpha)

# ===== RIGHT LEG (Swing phase, opposite) =====
r_hip = np.zeros_like(t)
r_knee = np.zeros_like(t)
r_ankle = np.zeros_like(t)

for i, time in enumerate(t):
    if time < half_period:
        # RIGHT LEG: SWING
        alpha = time / half_period
        r_hip[i] = hip_amp * 0.7 * np.sin(np.pi * alpha)
        r_knee[i] = -0.35 + knee_amp * np.cos(np.pi * alpha)
        r_ankle[i] = 0.2 + ankle_amp * np.cos(np.pi * alpha)
    else:
        # RIGHT LEG: STANCE - Push backward
        alpha = (time - half_period) / half_period
        r_hip[i] = -hip_amp * np.sin(np.pi * alpha)
        r_knee[i] = -0.6 - knee_amp * np.sin(np.pi * alpha)
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
print("MODERATE-AMPLITUDE FORWARD-BIASED WALKING GAIT")
print("=" * 80)
print(f"Gait period: {gait_period}s")
print(f"Sand grid: X=0.150 to 0.470m (length 0.32m)")
print(f"\nAmplitudes (MODERATE):")
print(f"  Hip swing: ±{hip_amp:.2f} rad (forward 70% of push)")
print(f"  Knee compression: {knee_amp:.2f} rad")
print(f"  Ankle angle: {ankle_amp:.2f} rad")

print(f"\nLEFT LEG (Stance 0-1.5s, Swing 1.5-3.0s)")
print(f"  Hip:   {l_hip.min():.4f} to {l_hip.max():.4f}")
print(f"  Knee:  {l_knee.min():.4f} to {l_knee.max():.4f}")
print(f"  Ankle: {l_ankle.min():.4f} to {l_ankle.max():.4f}")

print(f"\nExpected behavior: Steady forward walking, stays within sand bounds")
print("[+] Trajectories saved!")
