"""
Generate a PROPER walking gait with BOTH FEET ALWAYS ON SAND
Key: Keep feet in ground contact during entire cycle
"""
import numpy as np

gait_period = 3.0
half_period = gait_period / 2.0
t = np.linspace(0, gait_period, 100)

# ===== LEFT LEG (Stance 0-1.5s, Light swing 1.5-3.0s) =====
l_hip = np.zeros_like(t)
l_knee = np.zeros_like(t)
l_ankle = np.zeros_like(t)

for i, time in enumerate(t):
    if time < half_period:
        # LEFT LEG: STANCE - Push backward
        alpha = time / half_period
        l_hip[i] = -0.2 * np.sin(np.pi * alpha)  # Rotate backward
        l_knee[i] = -0.4 - 0.15 * np.sin(np.pi * alpha)  # Bent, pushing
        l_ankle[i] = -0.05  # Slight point down
    else:
        # LEFT LEG: LIGHT SWING - Minimal lifting
        alpha = (time - half_period) / half_period
        l_hip[i] = 0.15 * np.sin(np.pi * alpha)  # Small forward rotation
        l_knee[i] = -0.35 + 0.1 * np.cos(np.pi * alpha)  # Slightly bent, MINIMAL bend
        l_ankle[i] = 0.1 + 0.05 * np.cos(np.pi * alpha)  # Dorsiflexed

# ===== RIGHT LEG (Swing 0-1.5s, Stance 1.5-3.0s) =====
r_hip = np.zeros_like(t)
r_knee = np.zeros_like(t)
r_ankle = np.zeros_like(t)

for i, time in enumerate(t):
    if time < half_period:
        # RIGHT LEG: LIGHT SWING
        alpha = time / half_period
        r_hip[i] = 0.15 * np.sin(np.pi * alpha)
        r_knee[i] = -0.35 + 0.1 * np.cos(np.pi * alpha)
        r_ankle[i] = 0.1 + 0.05 * np.cos(np.pi * alpha)
    else:
        # RIGHT LEG: STANCE
        alpha = (time - half_period) / half_period
        r_hip[i] = -0.2 * np.sin(np.pi * alpha)
        r_knee[i] = -0.4 - 0.15 * np.sin(np.pi * alpha)
        r_ankle[i] = -0.05

# Save
np.save('ik_times.npy', t)
np.save('ik_left_hip.npy', l_hip)
np.save('ik_left_knee.npy', l_knee)
np.save('ik_left_ankle.npy', l_ankle)
np.save('ik_right_hip.npy', r_hip)
np.save('ik_right_knee.npy', r_knee)
np.save('ik_right_ankle.npy', r_ankle)

print("=" * 80)
print("GROUND-CONTACT WALKING GAIT (Both feet ON sand)")
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
print("  - Minimal knee lift during swing (stay on ground)")
print("  - Small hip rotation for stepping")
print("  - Both feet maintain contact with sand")
print("[+] Trajectories saved!")
