#!/usr/bin/env python3
"""
Generate aggressive forward-walking gait with asymmetric timing
Robot walks forward on sand with dominant push phase
"""

import numpy as np
import matplotlib.pyplot as plt

# Create an aggressive forward-walking gait
gait_period = 4.0
num_points = 100
time = np.linspace(0, gait_period, num_points)

# Normalize time to [0, 1]
t_norm = time / gait_period

# ============================================================================
# ASYMMETRIC GAIT - Longer push phase (0.6s), shorter swing (0.4s)
# ============================================================================

right_hip = np.zeros_like(time)
right_knee = np.zeros_like(time)
right_ankle = np.zeros_like(time)

left_hip = np.zeros_like(time)
left_knee = np.zeros_like(time)
left_ankle = np.zeros_like(time)

for i, t in enumerate(t_norm):
    if t < 0.55:
        # Longer phase (55%): Right leg support pushes hard, left leg swings
        phase = t / 0.55  # 0 to 1
        
        # Right leg: STRONG extension back for forward push
        right_hip[i] = 0.25 * np.sin(phase * np.pi)  # Stronger: 0 -> 0.25 -> 0
        right_knee[i] = -0.15 * (1 - np.cos(phase * np.pi)) / 2
        right_ankle[i] = 0.08 * np.sin(phase * np.pi)
        
        # Left leg: swing forward (quick)
        left_hip[i] = -0.20 * np.sin(phase * np.pi)  # 0 -> -0.20 -> 0
        left_knee[i] = -0.22 * np.sin(phase * np.pi)
        left_ankle[i] = 0.08 * np.sin(phase * np.pi)
        
    else:
        # Shorter phase (45%): Left leg support pushes, right leg swings back quickly
        phase = (t - 0.55) / 0.45  # 0 to 1
        
        # Left leg: STRONG extension back for forward push
        left_hip[i] = 0.25 * np.sin(phase * np.pi)  # Stronger: 0 -> 0.25 -> 0
        left_knee[i] = -0.15 * (1 - np.cos(phase * np.pi)) / 2
        left_ankle[i] = 0.08 * np.sin(phase * np.pi)
        
        # Right leg: swing forward (quick)
        right_hip[i] = -0.20 * np.sin(phase * np.pi)  # 0 -> -0.20 -> 0
        right_knee[i] = -0.22 * np.sin(phase * np.pi)
        right_ankle[i] = 0.08 * np.sin(phase * np.pi)

# Save the gait
np.save("ik_times.npy", time)
np.save("ik_left_hip.npy", left_hip)
np.save("ik_left_knee.npy", left_knee)
np.save("ik_left_ankle.npy", left_ankle)
np.save("ik_right_hip.npy", right_hip)
np.save("ik_right_knee.npy", right_knee)
np.save("ik_right_ankle.npy", right_ankle)

print("[+] Generated AGGRESSIVE FORWARD GAIT with asymmetric timing")
print(f"    Period: {gait_period}s")
print(f"    Points: {num_points}")
print(f"\nGait Analysis:")
print(f"    Right Hip:  min={np.min(right_hip):+.3f}, max={np.max(right_hip):+.3f}")
print(f"    Left Hip:   min={np.min(left_hip):+.3f}, max={np.max(left_hip):+.3f}")
print(f"\nPattern:")
print(f"    0.0-0.55s: Right leg STRONG push (0.25 rad), Left leg quick swing")
print(f"    0.55-1.0s: Left leg STRONG push (0.25 rad), Right leg quick swing")
print(f"    Result: Longer push phase + stronger push = forward motion dominates")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(time, left_hip, 'b-', label='Left Hip', linewidth=2)
axes[0].plot(time, right_hip, 'r-', label='Right Hip', linewidth=2)
axes[0].axvline(0.55, color='gray', linestyle='--', alpha=0.5)
axes[0].set_ylabel('Hip Angle (rad)')
axes[0].set_title('Aggressive Forward-Walking Gait (Asymmetric Timing)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, left_knee, 'b-', label='Left Knee', linewidth=2)
axes[1].plot(time, right_knee, 'r-', label='Right Knee', linewidth=2)
axes[1].axvline(0.55, color='gray', linestyle='--', alpha=0.5)
axes[1].set_ylabel('Knee Angle (rad)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(time, left_ankle, 'b-', label='Left Ankle', linewidth=2)
axes[2].plot(time, right_ankle, 'r-', label='Right Ankle', linewidth=2)
axes[2].axvline(0.55, color='gray', linestyle='--', alpha=0.5)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Ankle Angle (rad)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('aggressive_forward_gait.png', dpi=100, bbox_inches='tight')
print("\n[+] Saved gait visualization to aggressive_forward_gait.png")
plt.close()
