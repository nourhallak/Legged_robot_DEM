#!/usr/bin/env python3
"""
Generate a forward-walking gait with forward bias
Robot walks forward on sand with controlled, small steps
"""

import numpy as np
import matplotlib.pyplot as plt

# Create a forward-walking gait with forward bias
gait_period = 4.0  # Slower, more controlled
num_points = 100
time = np.linspace(0, gait_period, num_points)

# Normalize time to [0, 1]
t_norm = time / gait_period

# ============================================================================
# FORWARD-BIASED GAIT - Small steps with forward push
# ============================================================================

right_hip = np.zeros_like(time)
right_knee = np.zeros_like(time)
right_ankle = np.zeros_like(time)

left_hip = np.zeros_like(time)
left_knee = np.zeros_like(time)
left_ankle = np.zeros_like(time)

for i, t in enumerate(t_norm):
    if t < 0.5:
        # First half: Right leg support (pushes forward), left leg swing (forward lift)
        phase = t / 0.5  # 0 to 1
        
        # Right leg: extend back for forward push
        right_hip[i] = 0.15 * np.sin(phase * np.pi)  # 0 -> 0.15 -> 0
        right_knee[i] = -0.15 * (1 - np.cos(phase * np.pi)) / 2  # Slight bend
        right_ankle[i] = 0.05 * np.sin(phase * np.pi)
        
        # Left leg: swing forward (lifted, larger swing for forward motion)
        left_hip[i] = -0.22 * np.sin(phase * np.pi)  # Larger swing: 0 -> -0.22 -> 0
        left_knee[i] = -0.25 * np.sin(phase * np.pi)  # Higher knee lift
        left_ankle[i] = 0.10 * np.sin(phase * np.pi)
        
    else:
        # Second half: Left leg support, right leg swing (forward biased)
        phase = (t - 0.5) / 0.5  # 0 to 1
        
        # Left leg: extend back for forward push
        left_hip[i] = 0.15 * np.sin(phase * np.pi)  # 0 -> 0.15 -> 0
        left_knee[i] = -0.15 * (1 - np.cos(phase * np.pi)) / 2  # Slight bend
        left_ankle[i] = 0.05 * np.sin(phase * np.pi)
        
        # Right leg: swing forward (lifted, larger swing for forward motion)
        right_hip[i] = -0.22 * np.sin(phase * np.pi)  # Larger swing: 0 -> -0.22 -> 0
        right_knee[i] = -0.25 * np.sin(phase * np.pi)  # Higher knee lift
        right_ankle[i] = 0.10 * np.sin(phase * np.pi)

# Save the gait
np.save("ik_times.npy", time)
np.save("ik_left_hip.npy", left_hip)
np.save("ik_left_knee.npy", left_knee)
np.save("ik_left_ankle.npy", left_ankle)
np.save("ik_right_hip.npy", right_hip)
np.save("ik_right_knee.npy", right_knee)
np.save("ik_right_ankle.npy", right_ankle)

print("[+] Generated FORWARD-WALKING gait with sand contact")
print(f"    Period: {gait_period}s (controlled walking)")
print(f"    Points: {num_points}")
print(f"\nGait Analysis:")
print(f"    Right Hip:  min={np.min(right_hip):+.3f}, max={np.max(right_hip):+.3f}")
print(f"    Left Hip:   min={np.min(left_hip):+.3f}, max={np.max(left_hip):+.3f}")
print(f"    Right Knee: min={np.min(right_knee):+.3f}, max={np.max(right_knee):+.3f}")
print(f"    Left Knee:  min={np.min(left_knee):+.3f}, max={np.max(left_knee):+.3f}")
print(f"\nPattern:")
print(f"    0.0-0.5s: Right leg pushes (0.15 rad), Left leg swings forward (-0.22 rad)")
print(f"    0.5-1.0s: Left leg pushes (0.15 rad), Right leg swings forward (-0.22 rad)")
print(f"    Result: Forward-biased walking - robot walks forward on sand")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(time, left_hip, 'b-', label='Left Hip', linewidth=2)
axes[0].plot(time, right_hip, 'r-', label='Right Hip', linewidth=2)
axes[0].set_ylabel('Hip Angle (rad)')
axes[0].set_title('Forward-Walking Gait on Sand')
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
plt.savefig('forward_gait.png', dpi=100, bbox_inches='tight')
print("\n[+] Saved gait visualization to forward_gait.png")
plt.close()
