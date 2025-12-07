"""
Generate extended walking gait with multiple steps for visualization
"""
import numpy as np
from scipy.interpolate import interp1d

print("[+] Creating extended multi-step walking gait...\n")

# Original single gait cycle data
gait_period = 3.0  # seconds for one step pair

# Create a longer trajectory with multiple step cycles
num_cycles = 10  # 10 complete gait cycles = lots of stepping

# Time points for extended gait (10 cycles)
ik_times = np.linspace(0, num_cycles * gait_period, 500)

# Define a single step cycle (one complete gait)
single_cycle_time = np.linspace(0, gait_period, 50)

# LEFT LEG: Hip
# Stance phase (0-1.5s): -0.3 to 0 (retraction), Knee -0.85 to -0.28 (compression)
# Swing phase (1.5-3.0s): 0 to 0.2 (advancement), Knee -0.28 to -0.40 (minimal lift)
left_hip_cycle = np.concatenate([
    np.linspace(-0.30, 0.0, 25),      # Stance: retraction
    np.linspace(0.0, 0.2, 25)          # Swing: advancement
])

left_knee_cycle = np.concatenate([
    np.linspace(-0.85, -0.28, 25),     # Stance: compression
    np.linspace(-0.28, -0.40, 25)      # Swing: minimal lift (stays bent!)
])

left_ankle_cycle = np.concatenate([
    np.linspace(-0.15, -0.15, 25),     # Stance: push down
    np.linspace(-0.15, 0.15, 25)       # Swing: lift (minimal)
])

# RIGHT LEG: 180° out of phase with left
right_hip_cycle = np.concatenate([
    np.linspace(0.0, 0.2, 25),         # Swing phase (opposite to left)
    np.linspace(0.2, -0.30, 25)        # Swing to stance transition
])

right_knee_cycle = np.concatenate([
    np.linspace(-0.28, -0.40, 25),     # Swing: minimal lift
    np.linspace(-0.40, -0.85, 25)      # Transition to stance compression
])

right_ankle_cycle = np.concatenate([
    np.linspace(0.15, -0.15, 25),      # Swing to stance
    np.linspace(-0.15, -0.15, 25)      # Stance: push down
])

# Extend over multiple cycles
ik_left_hip = np.tile(left_hip_cycle, num_cycles)
ik_left_knee = np.tile(left_knee_cycle, num_cycles)
ik_left_ankle = np.tile(left_ankle_cycle, num_cycles)
ik_right_hip = np.tile(right_hip_cycle, num_cycles)
ik_right_knee = np.tile(right_knee_cycle, num_cycles)
ik_right_ankle = np.tile(right_ankle_cycle, num_cycles)

print(f"[+] Generated {num_cycles} complete gait cycles")
print(f"[+] Total trajectory duration: {ik_times[-1]:.1f}s")
print(f"[+] Time points: {len(ik_times)}")
print(f"[+] Data points per trajectory: {len(ik_left_hip)}")

# Save trajectories
np.save('ik_times.npy', ik_times)
np.save('ik_left_hip.npy', ik_left_hip)
np.save('ik_left_knee.npy', ik_left_knee)
np.save('ik_left_ankle.npy', ik_left_ankle)
np.save('ik_right_hip.npy', ik_right_hip)
np.save('ik_right_knee.npy', ik_right_knee)
np.save('ik_right_ankle.npy', ik_right_ankle)

print("\n[+] Trajectories saved!")
print(f"    - ik_times.npy: {len(ik_times)} time points")
print(f"    - ik_left_hip.npy: {len(ik_left_hip)} points")
print(f"    - ik_left_knee.npy: {len(ik_left_knee)} points")
print(f"    - ik_left_ankle.npy: {len(ik_left_ankle)} points")
print(f"    - ik_right_hip.npy: {len(ik_right_hip)} points")
print(f"    - ik_right_knee.npy: {len(ik_right_knee)} points")
print(f"    - ik_right_ankle.npy: {len(ik_right_ankle)} points")

print("\n[+] Left leg joints:")
print(f"    Hip:   {ik_left_hip[0]:.3f} → {ik_left_hip[-1]:.3f} rad")
print(f"    Knee:  {ik_left_knee[0]:.3f} → {ik_left_knee[-1]:.3f} rad")
print(f"    Ankle: {ik_left_ankle[0]:.3f} → {ik_left_ankle[-1]:.3f} rad")

print("\n[+] Right leg joints:")
print(f"    Hip:   {ik_right_hip[0]:.3f} → {ik_right_hip[-1]:.3f} rad")
print(f"    Knee:  {ik_right_knee[0]:.3f} → {ik_right_knee[-1]:.3f} rad")
print(f"    Ankle: {ik_right_ankle[0]:.3f} → {ik_right_ankle[-1]:.3f} rad")

print("\n[✓] Ready for walking visualization with extended gait!")
