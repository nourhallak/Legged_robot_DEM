#!/usr/bin/env python3
"""
Generate FORWARD-WALKING GAIT with momentum bias
Swing leg moves further forward than stance leg pushes back = net forward motion
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
    if t < 0.5:
        # First 50%: Right leg STANCE (push back 0.12 rad), Left leg SWING (forward -0.30 rad)
        phase = t / 0.5
        
        # Right leg STANCE: small backward push
        right_hip[i] = 0.12 * np.sin(phase * np.pi)  # Push back: 0->0.12->0
        right_knee[i] = -0.08 * (1 - np.cos(phase * np.pi))
        right_ankle[i] = 0.04 * np.sin(phase * np.pi)
        
        # Left leg SWING: larger forward movement (bias toward forward)
        left_hip[i] = -0.30 * np.sin(phase * np.pi)  # Swing forward: 0->-0.30->0 (LARGER)
        left_knee[i] = -0.40 * np.sin(phase * np.pi)  # Higher knee lift
        left_ankle[i] = 0.14 * np.sin(phase * np.pi)
        
    else:
        # Second 50%: Left leg STANCE (push back 0.12 rad), Right leg SWING (forward -0.30 rad)
        phase = (t - 0.5) / 0.5
        
        # Left leg STANCE: small backward push
        left_hip[i] = 0.12 * np.sin(phase * np.pi)  # Push back: 0->0.12->0
        left_knee[i] = -0.08 * (1 - np.cos(phase * np.pi))
        left_ankle[i] = 0.04 * np.sin(phase * np.pi)
        
        # Right leg SWING: larger forward movement (bias toward forward)
        right_hip[i] = -0.30 * np.sin(phase * np.pi)  # Swing forward: 0->-0.30->0 (LARGER)
        right_knee[i] = -0.40 * np.sin(phase * np.pi)  # Higher knee lift
        right_ankle[i] = 0.14 * np.sin(phase * np.pi)

# Save the gait
np.save("ik_times.npy", time)
np.save("ik_left_hip.npy", left_hip)
np.save("ik_left_knee.npy", left_knee)
np.save("ik_left_ankle.npy", left_ankle)
np.save("ik_right_hip.npy", right_hip)
np.save("ik_right_knee.npy", right_knee)
np.save("ik_right_ankle.npy", right_ankle)

print("[+] Generated FORWARD-BIASED WALKING GAIT")
print(f"    Period: {gait_period}s")
print(f"    Points: {num_points}")
print(f"\nGait Analysis:")
print(f"    Right Hip:  min={np.min(right_hip):+.3f}, max={np.max(right_hip):+.3f}")
print(f"    Left Hip:   min={np.min(left_hip):+.3f}, max={np.max(left_hip):+.3f}")
print(f"\nGait Design:")
print(f"    Stance leg push: 0.12 rad (backward)")
print(f"    Swing leg swing: 0.30 rad (forward) - LARGER for forward bias")
print(f"    Ratio: Swing/Push = 0.30/0.12 = 2.5x")
print(f"    Result: Net forward momentum accumulates each step")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(time, left_hip, 'b-', label='Left Hip', linewidth=2.5)
axes[0].plot(time, right_hip, 'r-', label='Right Hip', linewidth=2.5)
axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0].fill_between([0, 0.5], -0.4, 0.2, alpha=0.1, color='red', label='Right Stance (small push)')
axes[0].fill_between([0.5, 1], -0.4, 0.2, alpha=0.1, color='blue', label='Left Stance (small push)')
axes[0].set_ylabel('Hip Angle (rad)', fontsize=11)
axes[0].set_title('Forward-Biased Walking Gait (Swing > Push)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9, loc='upper right')
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
plt.savefig('forward_biased_gait.png', dpi=100, bbox_inches='tight')
print("\n[+] Saved gait visualization to forward_biased_gait.png")
plt.close()
