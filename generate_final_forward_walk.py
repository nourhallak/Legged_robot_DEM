#!/usr/bin/env python3
"""
Generate forward walking gait - SIGN FLIPPED for positive X motion
"""

import numpy as np
import matplotlib.pyplot as plt

gait_period = 3.0
num_points = 100
time = np.linspace(0, gait_period, num_points)

t_norm = time / gait_period

right_hip = np.zeros_like(time)
right_knee = np.zeros_like(time)
right_ankle = np.zeros_like(time)

left_hip = np.zeros_like(time)
left_knee = np.zeros_like(time)
left_ankle = np.zeros_like(time)

for i, t in enumerate(t_norm):
    if t < 0.33:
        # Phase 1: Left leg forward push, Right leg starting swing
        phase = t / 0.33
        # NEGATIVE hip = forward motion
        left_hip[i] = -0.40 * np.sin(phase * np.pi)  # Forward push: 0->-0.40->0
        left_knee[i] = -0.10 * (1 - np.cos(phase * np.pi))
        left_ankle[i] = 0.06 * np.sin(phase * np.pi)
        
        right_hip[i] = 0
        right_knee[i] = 0
        right_ankle[i] = 0
        
    elif t < 0.67:
        # Phase 2: Left leg return, Right leg SWING (forward, longer)
        phase = (t - 0.33) / 0.34
        
        left_hip[i] = -0.40 * np.cos(phase * np.pi)  # Returning: -0.40->0
        left_knee[i] = -0.10 * np.sin(phase * np.pi)**2
        left_ankle[i] = 0.06 * np.cos(phase * np.pi)
        
        # NEGATIVE = forward, larger swing for forward bias
        right_hip[i] = 0.45 * np.sin(phase * np.pi)  # Swing forward: 0->0.45->0 (STRONG)
        right_knee[i] = -0.50 * np.sin(phase * np.pi)  # High knee
        right_ankle[i] = 0.18 * np.sin(phase * np.pi)
        
    else:
        # Phase 3: Right leg forward push, Left leg starting swing
        phase = (t - 0.67) / 0.33
        right_hip[i] = 0.40 * np.sin(phase * np.pi)  # Forward push: 0->0.40->0
        right_knee[i] = -0.10 * (1 - np.cos(phase * np.pi))
        right_ankle[i] = 0.06 * np.sin(phase * np.pi)
        
        left_hip[i] = 0
        left_knee[i] = 0
        left_ankle[i] = 0

# Save the gait
np.save("ik_times.npy", time)
np.save("ik_left_hip.npy", left_hip)
np.save("ik_left_knee.npy", left_knee)
np.save("ik_left_ankle.npy", left_ankle)
np.save("ik_right_hip.npy", right_hip)
np.save("ik_right_knee.npy", right_knee)
np.save("ik_right_ankle.npy", right_ankle)

print("[+] Generated FORWARD WALKING GAIT - SIGN CORRECTED")
print(f"    Period: {gait_period}s")
print(f"    Points: {num_points}")
print(f"\nGait Design (Sign-Flipped):")
print(f"    Negative hip = forward push")
print(f"    Positive hip = return/swing")
print(f"\nExpected Result:")
print(f"    Robot walks FORWARD (+X direction)")
print(f"    All sand particles stay engaged")
print(f"    Stable motion on sand bed")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(time, left_hip, 'b-', label='Left Hip', linewidth=2.5)
axes[0].plot(time, right_hip, 'r-', label='Right Hip', linewidth=2.5)
axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0].set_ylabel('Hip Angle (rad)', fontsize=11)
axes[0].set_title('Forward Walking Gait - Corrected Signs', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, left_knee, 'b-', label='Left Knee', linewidth=2.5)
axes[1].plot(time, right_knee, 'r-', label='Right Knee', linewidth=2.5)
axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1].set_ylabel('Knee Angle (rad)', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

axes[2].plot(time, left_ankle, 'b-', label='Left Ankle', linewidth=2.5)
axes[2].plot(time, right_ankle, 'r-', label='Right Ankle', linewidth=2.5)
axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[2].set_xlabel('Time (s)', fontsize=11)
axes[2].set_ylabel('Ankle Angle (rad)', fontsize=11)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('forward_walking_final.png', dpi=100, bbox_inches='tight')
print("\n[+] Saved gait visualization to forward_walking_final.png")
plt.close()
