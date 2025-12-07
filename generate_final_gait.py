"""
Generate the original FORWARD-BIASED WALKING GAIT
(Now robot has boundary walls, so it won't escape)
"""
import numpy as np

gait_period = 3.0
half_period = gait_period / 2.0
t = np.linspace(0, gait_period, 100)

# ORIGINAL AMPLITUDES - good forward progress
hip_amp = 0.3
knee_amp = 0.2
ankle_amp = 0.2

# ===== LEFT LEG (Push/Stance phase) =====
l_hip = np.zeros_like(t)
l_knee = np.zeros_like(t)
l_ankle = np.zeros_like(t)

for i, time in enumerate(t):
    if time < half_period:
        # LEFT LEG: STANCE - Push backward
        alpha = time / half_period
        l_hip[i] = -hip_amp * np.sin(np.pi * alpha)
        l_knee[i] = -0.8 - knee_amp * np.sin(np.pi * alpha)
        l_ankle[i] = -0.2
    else:
        # LEFT LEG: SWING - Bring forward
        alpha = (time - half_period) / half_period
        l_hip[i] = hip_amp * 0.65 * np.sin(np.pi * alpha)
        l_knee[i] = -0.4 + knee_amp * np.cos(np.pi * alpha)
        l_ankle[i] = 0.3 + ankle_amp * 0.5 * np.cos(np.pi * alpha)

# ===== RIGHT LEG (Swing phase, opposite) =====
r_hip = np.zeros_like(t)
r_knee = np.zeros_like(t)
r_ankle = np.zeros_like(t)

for i, time in enumerate(t):
    if time < half_period:
        # RIGHT LEG: SWING
        alpha = time / half_period
        r_hip[i] = hip_amp * 0.65 * np.sin(np.pi * alpha)
        r_knee[i] = -0.4 + knee_amp * np.cos(np.pi * alpha)
        r_ankle[i] = 0.3 + ankle_amp * 0.5 * np.cos(np.pi * alpha)
    else:
        # RIGHT LEG: STANCE - Push backward
        alpha = (time - half_period) / half_period
        r_hip[i] = -hip_amp * np.sin(np.pi * alpha)
        r_knee[i] = -0.8 - knee_amp * np.sin(np.pi * alpha)
        r_ankle[i] = -0.2

# Save
np.save('ik_times.npy', t)
np.save('ik_left_hip.npy', l_hip)
np.save('ik_left_knee.npy', l_knee)
np.save('ik_left_ankle.npy', l_ankle)
np.save('ik_right_hip.npy', r_hip)
np.save('ik_right_knee.npy', r_knee)
np.save('ik_right_ankle.npy', r_ankle)

print("=" * 80)
print("FORWARD-BIASED WALKING GAIT (WITH BOUNDARY WALLS)")
print("=" * 80)
print(f"Gait period: {gait_period}s")
print(f"Sand grid: X=0.150 to 0.470m (with walls at 0.140 and 0.480m)")
print(f"Robot will now STAY ON SAND thanks to boundary walls")
print("[+] Trajectories saved!")
