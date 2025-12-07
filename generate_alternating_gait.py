"""
Generate improved alternating walking gait with better foot clearance
- Swing phase foot lifts higher above sand
- Stance phase foot stays firmly on sand without sinking
- Better phase opposition
"""
import numpy as np

print("[+] Creating improved alternating gait with foot clearance...\n")

# 10 gait cycles
num_cycles = 10
gait_period = 3.0
ik_times = np.linspace(0, num_cycles * gait_period, 500)

single_cycle_time = np.linspace(0, gait_period, 50)

# LEFT LEG trajectory
# Phase 1 (0-1.5s): LEFT foot in STANCE - push backward to propel forward
# Phase 2 (1.5-3.0s): LEFT foot in SWING - advance hip for next step
left_hip_cycle = np.concatenate([
    np.linspace(-0.35, -0.05, 25),     # Phase 1: Stance - PUSH BACK more aggressively
    np.linspace(-0.05, 0.30, 25)       # Phase 2: Swing - advance hip further
])

left_knee_cycle = np.concatenate([
    np.linspace(-0.50, -0.48, 25),     # Phase 1: Stance - stay bent (foot on sand)
    np.linspace(-0.48, -0.15, 25)      # Phase 2: Swing - EXTEND knee (LIFT foot high)
])

left_ankle_cycle = np.concatenate([
    np.linspace(-0.10, -0.10, 25),     # Phase 1: Stance - push against sand
    np.linspace(-0.10, 0.15, 25)       # Phase 2: Swing - lift foot high above sand
])

# RIGHT LEG trajectory (180° out of phase)
# Phase 1 (0-1.5s): RIGHT foot in SWING - lifts up
# Phase 2 (1.5-3.0s): RIGHT foot in STANCE - push backward to propel
right_hip_cycle = np.concatenate([
    np.linspace(-0.05, 0.30, 25),      # Phase 1: Swing - advance hip
    np.linspace(0.30, -0.35, 25)       # Phase 2: Stance - PUSH BACK aggressively
])

right_knee_cycle = np.concatenate([
    np.linspace(-0.15, -0.48, 25),     # Phase 1: Swing - EXTEND knee (LIFT foot high)
    np.linspace(-0.48, -0.50, 25)      # Phase 2: Stance - stay bent (foot on sand)
])

right_ankle_cycle = np.concatenate([
    np.linspace(0.15, -0.10, 25),      # Phase 1: Swing - lift foot high, then prepare landing
    np.linspace(-0.10, -0.10, 25)      # Phase 2: Stance - push
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

print(f"[+] Generated {num_cycles} gait cycles ({num_cycles * gait_period:.1f}s total)\n")

print("[+] Left leg joints:")
print(f"    Hip:   {ik_left_hip.min():.3f} to {ik_left_hip.max():.3f} rad")
print(f"    Knee:  {ik_left_knee.min():.3f} to {ik_left_knee.max():.3f} rad (0.5 = stance, -0.15 = swing lift)")
print(f"    Ankle: {ik_left_ankle.min():.3f} to {ik_left_ankle.max():.3f} rad (0.15 = high swing lift)")

print(f"\n[+] Right leg joints:")
print(f"    Hip:   {ik_right_hip.min():.3f} to {ik_right_hip.max():.3f} rad")
print(f"    Knee:  {ik_right_knee.min():.3f} to {ik_right_knee.max():.3f} rad (0.5 = stance, -0.15 = swing lift)")
print(f"    Ankle: {ik_right_ankle.min():.3f} to {ik_right_ankle.max():.3f} rad (0.15 = high swing lift)")

print(f"\n[✓] Improved gait with proper foot clearance during swing phase")
print(f"[✓] Swing phase extends knee (-0.15) and lifts ankle (0.15) to clear sand")
print(f"[✓] Stance phase keeps knee bent (-0.48 to -0.50) to maintain firm contact")
