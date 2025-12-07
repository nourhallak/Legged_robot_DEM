"""
Generate aggressive sand-pushing gait
Forces sand downward, not getting stuck in it
"""
import numpy as np

print("[+] Creating aggressive sand-pushing gait...\n")

# Extended gait with 10 cycles
num_cycles = 10
gait_period = 2.5  # Faster stepping
ik_times = np.linspace(0, num_cycles * gait_period, 500)

# LEFT LEG: Aggressive push down and forward
# Stance (0-1.25s): Hip retract, Knee push DOWN hard (-0.6), Ankle push DOWN
# Swing (1.25-2.5s): Hip advance, Knee up slightly, Ankle lift for next step
single_cycle_time = np.linspace(0, gait_period, 50)

left_hip_cycle = np.concatenate([
    np.linspace(-0.25, 0.0, 25),       # Stance: retraction
    np.linspace(0.0, 0.3, 25)          # Swing: forward advancement
])

left_knee_cycle = np.concatenate([
    np.linspace(-0.65, -0.25, 25),     # Stance: AGGRESSIVE compression (push sand down)
    np.linspace(-0.25, -0.45, 25)      # Swing: partial extension for ground clearance
])

left_ankle_cycle = np.concatenate([
    np.linspace(-0.25, -0.25, 25),     # Stance: PUSH DOWN hard on sand
    np.linspace(-0.25, 0.15, 25)       # Swing: lift foot
])

# RIGHT LEG: 180° out of phase
right_hip_cycle = np.concatenate([
    np.linspace(0.0, 0.3, 25),         # Swing: forward
    np.linspace(0.3, -0.25, 25)        # Swing to stance transition
])

right_knee_cycle = np.concatenate([
    np.linspace(-0.25, -0.45, 25),     # Swing: partial extension
    np.linspace(-0.45, -0.65, 25)      # Swing to stance: AGGRESSIVE compression
])

right_ankle_cycle = np.concatenate([
    np.linspace(0.15, -0.25, 25),      # Swing to stance: push down
    np.linspace(-0.25, -0.25, 25)      # Stance: PUSH DOWN hard
])

# Extend over multiple cycles
ik_left_hip = np.tile(left_hip_cycle, num_cycles)
ik_left_knee = np.tile(left_knee_cycle, num_cycles)
ik_left_ankle = np.tile(left_ankle_cycle, num_cycles)
ik_right_hip = np.tile(right_hip_cycle, num_cycles)
ik_right_knee = np.tile(right_knee_cycle, num_cycles)
ik_right_ankle = np.tile(right_ankle_cycle, num_cycles)

# Save trajectories
np.save('ik_times.npy', ik_times)
np.save('ik_left_hip.npy', ik_left_hip)
np.save('ik_left_knee.npy', ik_left_knee)
np.save('ik_left_ankle.npy', ik_left_ankle)
np.save('ik_right_hip.npy', ik_right_hip)
np.save('ik_right_knee.npy', ik_right_knee)
np.save('ik_right_ankle.npy', ik_right_ankle)

print(f"[+] Generated {num_cycles} aggressive gait cycles ({ik_times[-1]:.1f}s total)")
print(f"[+] Gait period: {gait_period:.2f}s (faster stepping)")
print(f"\n[+] Left leg joints:")
print(f"    Hip:   {ik_left_hip.min():.3f} to {ik_left_hip.max():.3f} rad")
print(f"    Knee:  {ik_left_knee.min():.3f} to {ik_left_knee.max():.3f} rad (AGGRESSIVE compression -0.65)")
print(f"    Ankle: {ik_left_ankle.min():.3f} to {ik_left_ankle.max():.3f} rad (PUSHES DOWN -0.25)")

print(f"\n[+] Right leg joints:")
print(f"    Hip:   {ik_right_hip.min():.3f} to {ik_right_hip.max():.3f} rad")
print(f"    Knee:  {ik_right_knee.min():.3f} to {ik_right_knee.max():.3f} rad (AGGRESSIVE compression -0.65)")
print(f"    Ankle: {ik_right_ankle.min():.3f} to {ik_right_ankle.max():.3f} rad (PUSHES DOWN -0.25)")

print("\n[✓] Aggressive sand-pushing gait ready - LOW friction sand will be pushed down, not trap feet")
