"""
Generate AGGRESSIVE stepping gait - BOTH FEET ON SAND + FORWARD MOTION
Increase knee compression for more pushing force
"""
import numpy as np

gait_period = 3.0
half_period = gait_period / 2.0
t = np.linspace(0, gait_period, 100)

# ===== LEFT LEG =====
l_hip = np.zeros_like(t)
l_knee = np.zeros_like(t)
l_ankle = np.zeros_like(t)

for i, time in enumerate(t):
    if time < half_period:
        # LEFT LEG: AGGRESSIVE STANCE - Push hard backward
        alpha = time / half_period
        l_hip[i] = -0.3 * np.sin(np.pi * alpha)  # Stronger hip rotation
        l_knee[i] = -0.6 - 0.25 * np.sin(np.pi * alpha)  # DEEPER knee compression
        l_ankle[i] = -0.15  # More aggressive point down
    else:
        # LEFT LEG: SWING - Minimal lift but ready for push
        alpha = (time - half_period) / half_period
        l_hip[i] = 0.2 * np.sin(np.pi * alpha)  # Stronger swing
        l_knee[i] = -0.4 + 0.12 * np.cos(np.pi * alpha)  # Still stays bent, minimal lift
        l_ankle[i] = 0.1 + 0.05 * np.cos(np.pi * alpha)

# ===== RIGHT LEG =====
r_hip = np.zeros_like(t)
r_knee = np.zeros_like(t)
r_ankle = np.zeros_like(t)

for i, time in enumerate(t):
    if time < half_period:
        # RIGHT LEG: SWING
        alpha = time / half_period
        r_hip[i] = 0.2 * np.sin(np.pi * alpha)
        r_knee[i] = -0.4 + 0.12 * np.cos(np.pi * alpha)
        r_ankle[i] = 0.1 + 0.05 * np.cos(np.pi * alpha)
    else:
        # RIGHT LEG: AGGRESSIVE STANCE
        alpha = (time - half_period) / half_period
        r_hip[i] = -0.3 * np.sin(np.pi * alpha)
        r_knee[i] = -0.6 - 0.25 * np.sin(np.pi * alpha)
        r_ankle[i] = -0.15

# Save
np.save('ik_times.npy', t)
np.save('ik_left_hip.npy', l_hip)
np.save('ik_left_knee.npy', l_knee)
np.save('ik_left_ankle.npy', l_ankle)
np.save('ik_right_hip.npy', r_hip)
np.save('ik_right_knee.npy', r_knee)
np.save('ik_right_ankle.npy', r_ankle)

print("=" * 80)
print("AGGRESSIVE STEPPING GAIT (Both feet ON sand + Forward motion)")
print("=" * 80)
print(f"Gait period: {gait_period}s\n")

print("LEFT LEG:")
print(f"  Hip:   {l_hip.min():.4f} to {l_hip.max():.4f}")
print(f"  Knee:  {l_knee.min():.4f} to {l_knee.max():.4f}")
print(f"  Ankle: {l_ankle.min():.4f} to {l_ankle.max():.4f}")

print("\nRIGHT LEG:")
print(f"  Hip:   {r_hip.min():.4f} to {r_hip.max():.4f}")
print(f"  Knee:  {r_knee.min():.4f} to {r_knee.max():.4f}")
print(f"  Ankle: {r_ankle.min():.4f} to {r_ankle.max():.4f}")

print("\nKey features:")
print("  - DEEP knee compression during stance (strong push)")
print("  - Aggressive hip rotation for stepping")
print("  - Minimal knee lift during swing (feet stay near ground)")
print("  - Both feet maintain contact with sand")
print("[+] Trajectories saved!")
