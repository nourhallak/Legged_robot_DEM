#!/usr/bin/env python3
"""
Generate a STABLE WALKING GAIT with proper weight distribution
Legs support body alternately while moving forward predictably
"""

import numpy as np
import matplotlib.pyplot as plt

gait_period = 3.0  # Slower for stability
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
        # First 50%: Right leg STANCE (supporting weight), Left leg SWING (moving forward)
        phase = t / 0.5
        
        # Right leg STANCE: mostly straight, slight backward angle for propulsion
        right_hip[i] = 0.15 * np.sin(phase * np.pi)  # Gentle backward push: 0->0.15->0
        right_knee[i] = -0.08 * (1 - np.cos(phase * np.pi))  # Slight flex when supporting
        right_ankle[i] = 0.05 * np.sin(phase * np.pi)
        
        # Left leg SWING: lifted and swung forward
        left_hip[i] = -0.25 * np.sin(phase * np.pi)  # Forward swing: 0->-0.25->0
        left_knee[i] = -0.35 * np.sin(phase * np.pi)  # Knee lift for clearance
        left_ankle[i] = 0.12 * np.sin(phase * np.pi)
        
    else:
        # Second 50%: Left leg STANCE (supporting weight), Right leg SWING (moving forward)
        phase = (t - 0.5) / 0.5
        
        # Left leg STANCE: mostly straight, slight backward angle for propulsion
        left_hip[i] = 0.15 * np.sin(phase * np.pi)  # Gentle backward push: 0->0.15->0
        left_knee[i] = -0.08 * (1 - np.cos(phase * np.pi))  # Slight flex when supporting
        left_ankle[i] = 0.05 * np.sin(phase * np.pi)
        
        # Right leg SWING: lifted and swung forward
        right_hip[i] = -0.25 * np.sin(phase * np.pi)  # Forward swing: 0->-0.25->0
        right_knee[i] = -0.35 * np.sin(phase * np.pi)  # Knee lift for clearance
        right_ankle[i] = 0.12 * np.sin(phase * np.pi)

# Save the gait
np.save("ik_times.npy", time)
np.save("ik_left_hip.npy", left_hip)
np.save("ik_left_knee.npy", left_knee)
np.save("ik_left_ankle.npy", left_ankle)
np.save("ik_right_hip.npy", right_hip)
np.save("ik_right_knee.npy", right_knee)
np.save("ik_right_ankle.npy", right_ankle)

print("[+] Generated STABLE WALKING GAIT with proper weight distribution")
print(f"    Period: {gait_period}s (slow, stable)")
print(f"    Points: {num_points}")
print(f"\nGait Analysis:")
print(f"    Right Hip:  min={np.min(right_hip):+.3f}, max={np.max(right_hip):+.3f}")
print(f"    Left Hip:   min={np.min(left_hip):+.3f}, max={np.max(left_hip):+.3f}")
print(f"    Right Knee: min={np.min(right_knee):+.3f}, max={np.max(right_knee):+.3f}")
print(f"    Left Knee:  min={np.min(left_knee):+.3f}, max={np.max(left_knee):+.3f}")
print(f"\nGait Phases:")
print(f"    0.0-0.5s: Right leg STANCE (push 0.15 rad), Left leg SWING (forward -0.25 rad)")
print(f"    0.5-1.0s: Left leg STANCE (push 0.15 rad), Right leg SWING (forward -0.25 rad)")
print(f"    Result: Stable alternating stance/swing for predictable forward motion")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(time, left_hip, 'b-', label='Left Hip', linewidth=2.5)
axes[0].plot(time, right_hip, 'r-', label='Right Hip', linewidth=2.5)
axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0].fill_between([0, 0.5], -0.4, 0.4, alpha=0.1, color='red', label='Right Stance')
axes[0].fill_between([0.5, 1], -0.4, 0.4, alpha=0.1, color='blue', label='Left Stance')
axes[0].set_ylabel('Hip Angle (rad)', fontsize=11)
axes[0].set_title('STABLE Walking Gait - Stance/Swing Phases', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10, loc='upper right')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 1)

axes[1].plot(time, left_knee, 'b-', label='Left Knee', linewidth=2.5)
axes[1].plot(time, right_knee, 'r-', label='Right Knee', linewidth=2.5)
axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1].fill_between([0, 0.5], -0.5, 0.1, alpha=0.1, color='red')
axes[1].fill_between([0.5, 1], -0.5, 0.1, alpha=0.1, color='blue')
axes[1].set_ylabel('Knee Angle (rad)', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 1)

axes[2].plot(time, left_ankle, 'b-', label='Left Ankle', linewidth=2.5)
axes[2].plot(time, right_ankle, 'r-', label='Right Ankle', linewidth=2.5)
axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[2].fill_between([0, 0.5], -0.2, 0.2, alpha=0.1, color='red')
axes[2].fill_between([0.5, 1], -0.2, 0.2, alpha=0.1, color='blue')
axes[2].set_xlabel('Time (normalized)', fontsize=11)
axes[2].set_ylabel('Ankle Angle (rad)', fontsize=11)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, 1)

plt.tight_layout()
plt.savefig('stable_stance_swing_gait.png', dpi=100, bbox_inches='tight')
print("\n[+] Saved gait visualization to stable_stance_swing_gait.png")
plt.close()
