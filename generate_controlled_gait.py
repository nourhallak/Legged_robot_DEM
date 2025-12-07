#!/usr/bin/env python3
"""
Generate controlled forward-walking gait - slower, stays on sand
Robot walks forward at controlled speed with maintained contact
"""

import numpy as np
import matplotlib.pyplot as plt

gait_period = 5.0  # Even slower
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
    if t < 0.55:
        # Longer phase: Right leg support, left leg swing
        phase = t / 0.55
        
        # Right leg: REDUCED push (0.12 rad instead of 0.25)
        right_hip[i] = 0.12 * np.sin(phase * np.pi)  # Much smaller: 0 -> 0.12 -> 0
        right_knee[i] = -0.12 * (1 - np.cos(phase * np.pi)) / 2
        right_ankle[i] = 0.05 * np.sin(phase * np.pi)
        
        # Left leg: swing forward (small)
        left_hip[i] = -0.14 * np.sin(phase * np.pi)  # 0 -> -0.14 -> 0
        left_knee[i] = -0.16 * np.sin(phase * np.pi)
        left_ankle[i] = 0.06 * np.sin(phase * np.pi)
        
    else:
        # Shorter phase: Left leg support, right leg swing
        phase = (t - 0.55) / 0.45
        
        # Left leg: REDUCED push (0.12 rad)
        left_hip[i] = 0.12 * np.sin(phase * np.pi)  # Much smaller: 0 -> 0.12 -> 0
        left_knee[i] = -0.12 * (1 - np.cos(phase * np.pi)) / 2
        left_ankle[i] = 0.05 * np.sin(phase * np.pi)
        
        # Right leg: swing forward (small)
        right_hip[i] = -0.14 * np.sin(phase * np.pi)  # 0 -> -0.14 -> 0
        right_knee[i] = -0.16 * np.sin(phase * np.pi)
        right_ankle[i] = 0.06 * np.sin(phase * np.pi)

# Save the gait
np.save("ik_times.npy", time)
np.save("ik_left_hip.npy", left_hip)
np.save("ik_left_knee.npy", left_knee)
np.save("ik_left_ankle.npy", left_ankle)
np.save("ik_right_hip.npy", right_hip)
np.save("ik_right_knee.npy", right_knee)
np.save("ik_right_ankle.npy", right_ankle)

print("[+] Generated CONTROLLED FORWARD GAIT (slow, stays on sand)")
print(f"    Period: {gait_period}s (slower movement)")
print(f"    Points: {num_points}")
print(f"\nGait Analysis:")
print(f"    Right Hip:  min={np.min(right_hip):+.3f}, max={np.max(right_hip):+.3f}")
print(f"    Left Hip:   min={np.min(left_hip):+.3f}, max={np.max(left_hip):+.3f}")
print(f"\nPattern:")
print(f"    0.0-0.55s: Right leg push (0.12 rad), Left leg swing (-0.14 rad)")
print(f"    0.55-1.0s: Left leg push (0.12 rad), Right leg swing (-0.14 rad)")
print(f"    Result: Slow, controlled forward walking - robot stays on sand")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(time, left_hip, 'b-', label='Left Hip', linewidth=2)
axes[0].plot(time, right_hip, 'r-', label='Right Hip', linewidth=2)
axes[0].set_ylabel('Hip Angle (rad)')
axes[0].set_title('Controlled Forward-Walking Gait (Slow)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, left_knee, 'b-', label='Left Knee', linewidth=2)
axes[1].plot(time, right_knee, 'r-', label='Right Knee', linewidth=2)
axes[1].set_ylabel('Knee Angle (rad)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(time, left_ankle, 'b-', label='Left Ankle', linewidth=2)
axes[2].plot(time, right_ankle, 'r-', label='Right Ankle', linewidth=2)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Ankle Angle (rad)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('controlled_forward_gait.png', dpi=100, bbox_inches='tight')
print("\n[+] Saved gait visualization to controlled_forward_gait.png")
plt.close()
