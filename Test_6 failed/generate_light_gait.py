"""
Generate light-step walking gait with both feet on sand
Reduced compression to avoid sticking
"""
import numpy as np

print("[+] Creating light-step walking gait (both feet on sand)...\n")

# Extended gait with 10 cycles
num_cycles = 10
gait_period = 3.0
ik_times = np.linspace(0, num_cycles * gait_period, 500)

# LEFT LEG: Light stepping pattern
# Stance (0-1.5s): Hip retraction (-0.2 to 0), Knee slight bend (-0.5 to -0.3)
# Swing (1.5-3.0s): Hip advance (0 to 0.2), Knee minimal (stays at -0.35)
single_cycle_time = np.linspace(0, gait_period, 50)

left_hip_cycle = np.concatenate([
    np.linspace(-0.20, 0.0, 25),       # Stance: light retraction
    np.linspace(0.0, 0.2, 25)          # Swing: advancement
])

left_knee_cycle = np.concatenate([
    np.linspace(-0.50, -0.30, 25),     # Stance: light compression
    np.linspace(-0.30, -0.35, 25)      # Swing: stays bent (no lift)
])

left_ankle_cycle = np.concatenate([
    np.linspace(-0.10, -0.10, 25),     # Stance: slight push
    np.linspace(-0.10, 0.05, 25)       # Swing: minimal lift
])

# RIGHT LEG: 180° out of phase
right_hip_cycle = np.concatenate([
    np.linspace(0.0, 0.2, 25),         # Swing phase
    np.linspace(0.2, -0.20, 25)        # Transition to stance
])

right_knee_cycle = np.concatenate([
    np.linspace(-0.30, -0.35, 25),     # Swing: stays bent
    np.linspace(-0.35, -0.50, 25)      # Transition to stance
])

right_ankle_cycle = np.concatenate([
    np.linspace(0.05, -0.10, 25),      # Swing to stance
    np.linspace(-0.10, -0.10, 25)      # Stance: push
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

print(f"[+] Generated {num_cycles} gait cycles ({ik_times[-1]:.1f}s total)")
print(f"\n[+] Left leg joints:")
print(f"    Hip:   {ik_left_hip.min():.3f} to {ik_left_hip.max():.3f} rad")
print(f"    Knee:  {ik_left_knee.min():.3f} to {ik_left_knee.max():.3f} rad (light compression)")
print(f"    Ankle: {ik_left_ankle.min():.3f} to {ik_left_ankle.max():.3f} rad")

print(f"\n[+] Right leg joints:")
print(f"    Hip:   {ik_right_hip.min():.3f} to {ik_right_hip.max():.3f} rad")
print(f"    Knee:  {ik_right_knee.min():.3f} to {ik_right_knee.max():.3f} rad (light compression)")
print(f"    Ankle: {ik_right_ankle.min():.3f} to {ik_right_ankle.max():.3f} rad")

print("\n[✓] Trajectories saved with light stepping (both feet ON sand)")
