#!/usr/bin/env python3
"""
Generate a symmetric forward-walking gait where the legs push the robot forward
This uses a simple alternating step pattern
"""

import numpy as np
import matplotlib.pyplot as plt

# Create a symmetric gait that pushes forward
gait_period = 3.0
num_points = 100
time = np.linspace(0, gait_period, num_points)

# Normalize time to [0, 1]
t_norm = time / gait_period

# ============================================================================
# SYMMETRIC FORWARD GAIT PATTERN
# ============================================================================
# Right leg (0 to 0.5): swinging, left leg (0.5 to 1.0): swinging
# When leg swings back, it pushes the body forward

# Right leg pattern (first half of cycle)
right_hip = np.zeros_like(time)
right_knee = np.zeros_like(time)
right_ankle = np.zeros_like(time)

# Left leg pattern (opposite phase)
left_hip = np.zeros_like(time)
left_knee = np.zeros_like(time)
left_ankle = np.zeros_like(time)

for i, t in enumerate(t_norm):
    if t < 0.5:
        # First half: Right leg extends backward (pushes), left leg swings forward (lifts)
        phase = t / 0.5  # 0 to 1
        
        # Right leg: extend back (positive hip angle = back)
        right_hip[i] = 0.3 * np.sin(phase * np.pi)  # Extends back: 0 -> 0.3 -> 0
        right_knee[i] = -0.2 * (1 - np.cos(phase * np.pi)) / 2  # Slightly bent
        right_ankle[i] = 0.1 * np.sin(phase * np.pi)
        
        # Left leg: swing forward (negative hip angle = forward)
        left_hip[i] = -0.35 * np.sin(phase * np.pi)  # Swings forward: 0 -> -0.35 -> 0
        left_knee[i] = -0.35 * np.sin(phase * np.pi)  # Lift knee high
        left_ankle[i] = 0.15 * np.sin(phase * np.pi)
        
    else:
        # Second half: Left leg extends backward, right leg swings forward
        phase = (t - 0.5) / 0.5  # 0 to 1
        
        # Left leg: extend back
        left_hip[i] = 0.3 * np.sin(phase * np.pi)  # Extends back: 0 -> 0.3 -> 0
        left_knee[i] = -0.2 * (1 - np.cos(phase * np.pi)) / 2  # Slightly bent
        left_ankle[i] = 0.1 * np.sin(phase * np.pi)
        
        # Right leg: swing forward
        right_hip[i] = -0.35 * np.sin(phase * np.pi)  # Swings forward: 0 -> -0.35 -> 0
        right_knee[i] = -0.35 * np.sin(phase * np.pi)  # Lift knee high
        right_ankle[i] = 0.15 * np.sin(phase * np.pi)

# Save the gait
np.save("ik_times.npy", time)
np.save("ik_left_hip.npy", left_hip)
np.save("ik_left_knee.npy", left_knee)
np.save("ik_left_ankle.npy", left_ankle)
np.save("ik_right_hip.npy", right_hip)
np.save("ik_right_knee.npy", right_knee)
np.save("ik_right_ankle.npy", right_ankle)

print("[+] Generated symmetric forward-walking gait")
print(f"    Period: {gait_period}s")
print(f"    Points: {num_points}")
print(f"\nGait Analysis:")
print(f"    Right Hip:  min={np.min(right_hip):+.3f}, max={np.max(right_hip):+.3f}")
print(f"    Left Hip:   min={np.min(left_hip):+.3f}, max={np.max(left_hip):+.3f}")
print(f"    Right Knee: min={np.min(right_knee):+.3f}, max={np.max(right_knee):+.3f}")
print(f"    Left Knee:  min={np.min(left_knee):+.3f}, max={np.max(left_knee):+.3f}")
print(f"\nPattern:")
print(f"    0.0-0.5s: Right leg pushes (extends back), Left leg swings (lifts)")
print(f"    0.5-1.0s: Left leg pushes (extends back), Right leg swings (lifts)")
print(f"    Result: Symmetric alternating push pattern for forward motion")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(time, left_hip, 'b-', label='Left Hip', linewidth=2)
axes[0].plot(time, right_hip, 'r-', label='Right Hip', linewidth=2)
axes[0].set_ylabel('Hip Angle (rad)')
axes[0].set_title('Symmetric Forward-Walking Gait')
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
plt.savefig('symmetric_gait.png', dpi=100, bbox_inches='tight')
print("\n[+] Saved gait visualization to symmetric_gait.png")
plt.close()
