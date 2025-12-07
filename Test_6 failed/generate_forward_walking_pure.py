"""
Generate a pure forward walking gait (NO lateral Y motion)
Left leg: Step forward, right leg stays back
Right leg: Step forward, left leg stays back
"""
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Forward walking gait - stepping forward in X direction only
# Each leg steps alternately, NO Y motion

gait_period = 3.0  # 3 seconds per full cycle (left step + right step)
half_period = gait_period / 2.0

# Time points
t = np.linspace(0, gait_period, 100)

# ===== LEFT LEG =====
# Stance phase (0 to 1.5s): Left leg pushes backward
# Swing phase (1.5 to 3.0s): Left leg swings forward

# L-Hip: Forward stepping motion (X direction creates forward step)
l_hip = np.zeros_like(t)
for i, time in enumerate(t):
    if time < half_period:
        # Stance: hip back, pushing
        l_hip[i] = -0.15 * np.sin(np.pi * time / half_period)
    else:
        # Swing: hip forward
        alpha = (time - half_period) / half_period
        l_hip[i] = 0.15 * np.sin(np.pi * alpha)

# L-Knee: Bent in stance, extended in swing
l_knee = np.zeros_like(t)
for i, time in enumerate(t):
    if time < half_period:
        # Stance: knee bent for pushing
        l_knee[i] = -0.3 - 0.2 * np.sin(np.pi * time / half_period)**2
    else:
        # Swing: knee bent for clearance then extended
        alpha = (time - half_period) / half_period
        l_knee[i] = -0.3 + 0.2 * np.cos(2 * np.pi * alpha)

# L-Ankle: Flat in stance, dorsiflexed in swing
l_ankle = np.zeros_like(t)
for i, time in enumerate(t):
    if time < half_period:
        # Stance: ankle neutral
        l_ankle[i] = 0.15
    else:
        # Swing: ankle dorsiflexed
        alpha = (time - half_period) / half_period
        l_ankle[i] = 0.15 + 0.1 * np.cos(np.pi * alpha)

# ===== RIGHT LEG =====
# Offset by half period (180 degrees)

# R-Hip: Opposite phase
r_hip = np.zeros_like(t)
for i, time in enumerate(t):
    if time < half_period:
        # Swing: hip forward
        r_hip[i] = 0.15 * np.sin(np.pi * time / half_period)
    else:
        # Stance: hip back, pushing
        alpha = (time - half_period) / half_period
        r_hip[i] = -0.15 * np.sin(np.pi * alpha)

# R-Knee
r_knee = np.zeros_like(t)
for i, time in enumerate(t):
    if time < half_period:
        # Swing: knee bent for clearance
        r_knee[i] = -0.3 + 0.2 * np.cos(2 * np.pi * time / half_period)
    else:
        # Stance: knee bent for pushing
        alpha = (time - half_period) / half_period
        r_knee[i] = -0.3 - 0.2 * np.sin(np.pi * alpha)**2

# R-Ankle
r_ankle = np.zeros_like(t)
for i, time in enumerate(t):
    if time < half_period:
        # Swing: ankle dorsiflexed
        r_ankle[i] = 0.15 + 0.1 * np.cos(np.pi * time / half_period)
    else:
        # Stance: ankle neutral
        alpha = (time - half_period) / half_period
        r_ankle[i] = 0.15

# Save trajectories
np.save('ik_times.npy', t)
np.save('ik_left_hip.npy', l_hip)
np.save('ik_left_knee.npy', l_knee)
np.save('ik_left_ankle.npy', l_ankle)
np.save('ik_right_hip.npy', r_hip)
np.save('ik_right_knee.npy', r_knee)
np.save('ik_right_ankle.npy', r_ankle)

print("=" * 80)
print("FORWARD WALKING GAIT GENERATED")
print("=" * 80)
print(f"Gait period: {gait_period}s (half period: {half_period}s)")
print(f"Points per cycle: {len(t)}")
print(f"\nLeft Leg (L):")
print(f"  Hip range:   {l_hip.min():.4f} to {l_hip.max():.4f}")
print(f"  Knee range:  {l_knee.min():.4f} to {l_knee.max():.4f}")
print(f"  Ankle range: {l_ankle.min():.4f} to {l_ankle.max():.4f}")
print(f"\nRight Leg (R):")
print(f"  Hip range:   {r_hip.min():.4f} to {r_hip.max():.4f}")
print(f"  Knee range:  {r_knee.min():.4f} to {r_knee.max():.4f}")
print(f"  Ankle range: {r_ankle.min():.4f} to {r_ankle.max():.4f}")
print(f"\n[+] Trajectories saved!")

# Plot
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Left leg
axes[0, 0].plot(t, l_hip, 'b-', linewidth=2)
axes[0, 0].set_title('Left Hip (Forward Step)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(half_period, color='r', linestyle='--', alpha=0.5, label='Phase transition')
axes[0, 0].legend()

axes[1, 0].plot(t, l_knee, 'b-', linewidth=2)
axes[1, 0].set_title('Left Knee')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axvline(half_period, color='r', linestyle='--', alpha=0.5)

axes[2, 0].plot(t, l_ankle, 'b-', linewidth=2)
axes[2, 0].set_title('Left Ankle')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].axvline(half_period, color='r', linestyle='--', alpha=0.5)

# Right leg
axes[0, 1].plot(t, r_hip, 'r-', linewidth=2)
axes[0, 1].set_title('Right Hip (Forward Step, 180° offset)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(half_period, color='r', linestyle='--', alpha=0.5, label='Phase transition')
axes[0, 1].legend()

axes[1, 1].plot(t, r_knee, 'r-', linewidth=2)
axes[1, 1].set_title('Right Knee')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvline(half_period, color='r', linestyle='--', alpha=0.5)

axes[2, 1].plot(t, r_ankle, 'r-', linewidth=2)
axes[2, 1].set_title('Right Ankle')
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].axvline(half_period, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('forward_walking_gait.png', dpi=150, bbox_inches='tight')
print(f"[+] Plot saved to forward_walking_gait.png")
plt.close()
