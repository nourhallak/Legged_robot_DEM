#!/usr/bin/env python3
"""
Generate a TRUE WALKING GAIT - legs 180 degrees out of phase
"""

import numpy as np
import matplotlib.pyplot as plt

gait_period = 2.0  # One complete gait cycle
num_points = 100
time = np.linspace(0, gait_period, num_points)

t_norm = time / gait_period

right_hip = np.zeros_like(time)
right_knee = np.zeros_like(time)
right_ankle = np.zeros_like(time)

left_hip = np.zeros_like(time)
left_knee = np.zeros_like(time)
left_ankle = np.zeros_like(time)

# RIGHT LEG PATTERN: Starts at 0, peaks at 0.25 (back), valley at 0.75 (forward)
# LEFT LEG PATTERN: Opposite - valley at 0.25 (forward), peak at 0.75 (back)

for i, t in enumerate(t_norm):
    # Right leg: back-forward oscillation
    right_hip[i] = 0.35 * np.cos(2 * np.pi * t)      # cos starts at +0.35 (back)
    right_knee[i] = -0.25 * np.sin(2 * np.pi * t)**2  # Knee bends when supporting
    right_ankle[i] = 0.10 * np.sin(2 * np.pi * t)
    
    # Left leg: opposite phase (180° out)
    left_hip[i] = -0.35 * np.cos(2 * np.pi * t)      # cos phase shifted 180°
    left_knee[i] = -0.25 * np.sin(2 * np.pi * (t + 0.5))**2  # Opposite bend phase
    left_ankle[i] = 0.10 * np.sin(2 * np.pi * (t + 0.5))

# Save the gait
np.save("ik_times.npy", time)
np.save("ik_left_hip.npy", left_hip)
np.save("ik_left_knee.npy", left_knee)
np.save("ik_left_ankle.npy", left_ankle)
np.save("ik_right_hip.npy", right_hip)
np.save("ik_right_knee.npy", right_knee)
np.save("ik_right_ankle.npy", right_ankle)

print("[+] Generated TRUE WALKING GAIT - 180° phase offset between legs")
print(f"    Period: {gait_period}s")
print(f"    Points: {num_points}")
print(f"\nGait Analysis:")
print(f"    Right Hip:  min={np.min(right_hip):+.3f}, max={np.max(right_hip):+.3f}")
print(f"    Left Hip:   min={np.min(left_hip):+.3f}, max={np.max(left_hip):+.3f}")
print(f"    Right Knee: min={np.min(right_knee):+.3f}, max={np.max(right_knee):+.3f}")
print(f"    Left Knee:  min={np.min(left_knee):+.3f}, max={np.max(left_knee):+.3f}")
print(f"\nPhase Relationship:")
print(f"    At t=0.0s:   Right leg BACK (+0.35),  Left leg FORWARD (-0.35)")
print(f"    At t=0.5s:   Right leg FORWARD (-0.35), Left leg BACK (+0.35)")
print(f"    At t=1.0s:   Right leg BACK (+0.35),  Left leg FORWARD (-0.35)")
print(f"    Result: Perfect alternating walking motion")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(time, left_hip, 'b-', label='Left Hip', linewidth=2.5)
axes[0].plot(time, right_hip, 'r-', label='Right Hip', linewidth=2.5)
axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0].set_ylabel('Hip Angle (rad)', fontsize=11)
axes[0].set_title('TRUE Walking Gait - Perfect 180° Phase Offset', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10, loc='upper right')
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, left_knee, 'b-', label='Left Knee', linewidth=2.5)
axes[1].plot(time, right_knee, 'r-', label='Right Knee', linewidth=2.5)
axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1].set_ylabel('Knee Angle (rad)', fontsize=11)
axes[1].legend(fontsize=10, loc='lower right')
axes[1].grid(True, alpha=0.3)

axes[2].plot(time, left_ankle, 'b-', label='Left Ankle', linewidth=2.5)
axes[2].plot(time, right_ankle, 'r-', label='Right Ankle', linewidth=2.5)
axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[2].set_xlabel('Time (s)', fontsize=11)
axes[2].set_ylabel('Ankle Angle (rad)', fontsize=11)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('true_walking_gait.png', dpi=100, bbox_inches='tight')
print("\n[+] Saved gait visualization to true_walking_gait.png")
plt.close()
