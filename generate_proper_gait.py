#!/usr/bin/env python3
"""
Generate a proper walking gait with large, clear alternating leg movements
"""

import numpy as np
import matplotlib.pyplot as plt

gait_period = 2.0  # Fast walking
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
    if t < 0.5:
        # First half: Right leg back (support), Left leg forward (swing)
        phase = t / 0.5  # 0 to 1
        
        # Right leg BACK - strong extension
        right_hip[i] = 0.35 * np.sin(phase * np.pi)  # 0 -> 0.35 -> 0 (back)
        right_knee[i] = -0.20 * (1 - np.cos(phase * np.pi)) / 2  # Slight knee bend
        right_ankle[i] = 0.10 * np.sin(phase * np.pi)
        
        # Left leg FORWARD - swing
        left_hip[i] = -0.35 * np.sin(phase * np.pi)  # 0 -> -0.35 -> 0 (forward)
        left_knee[i] = -0.40 * np.sin(phase * np.pi)  # Lift knee
        left_ankle[i] = 0.15 * np.sin(phase * np.pi)
        
    else:
        # Second half: Left leg back (support), Right leg forward (swing)
        phase = (t - 0.5) / 0.5  # 0 to 1
        
        # Left leg BACK - strong extension
        left_hip[i] = 0.35 * np.sin(phase * np.pi)  # 0 -> 0.35 -> 0 (back)
        left_knee[i] = -0.20 * (1 - np.cos(phase * np.pi)) / 2  # Slight knee bend
        left_ankle[i] = 0.10 * np.sin(phase * np.pi)
        
        # Right leg FORWARD - swing
        right_hip[i] = -0.35 * np.sin(phase * np.pi)  # 0 -> -0.35 -> 0 (forward)
        right_knee[i] = -0.40 * np.sin(phase * np.pi)  # Lift knee
        right_ankle[i] = 0.15 * np.sin(phase * np.pi)

# Save the gait
np.save("ik_times.npy", time)
np.save("ik_left_hip.npy", left_hip)
np.save("ik_left_knee.npy", left_knee)
np.save("ik_left_ankle.npy", left_ankle)
np.save("ik_right_hip.npy", right_hip)
np.save("ik_right_knee.npy", right_knee)
np.save("ik_right_ankle.npy", right_ankle)

print("[+] Generated PROPER WALKING GAIT - Large alternating leg movements")
print(f"    Period: {gait_period}s (fast natural walking)")
print(f"    Points: {num_points}")
print(f"\nGait Analysis:")
print(f"    Right Hip:  min={np.min(right_hip):+.3f}, max={np.max(right_hip):+.3f}")
print(f"    Left Hip:   min={np.min(left_hip):+.3f}, max={np.max(left_hip):+.3f}")
print(f"    Right Knee: min={np.min(right_knee):+.3f}, max={np.max(right_knee):+.3f}")
print(f"    Left Knee:  min={np.min(left_knee):+.3f}, max={np.max(left_knee):+.3f}")
print(f"\nPattern:")
print(f"    0.0-0.5s: Right leg extends BACK (0.35 rad), Left leg swings FORWARD (-0.35 rad)")
print(f"    0.5-1.0s: Left leg extends BACK (0.35 rad), Right leg swings FORWARD (-0.35 rad)")
print(f"    Result: Clear alternating walking motion")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(time, left_hip, 'b-', label='Left Hip', linewidth=2.5)
axes[0].plot(time, right_hip, 'r-', label='Right Hip', linewidth=2.5)
axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0].set_ylabel('Hip Angle (rad)', fontsize=11)
axes[0].set_title('PROPER Walking Gait - Large Alternating Movements', fontsize=12, fontweight='bold')
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
plt.savefig('proper_walking_gait.png', dpi=100, bbox_inches='tight')
print("\n[+] Saved gait visualization to proper_walking_gait.png")
plt.close()
