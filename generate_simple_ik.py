#!/usr/bin/env python3
"""
Simple solution: use constant joint angles based on gait phase.
"""

import numpy as np

def generate_simple_walking_angles(duration=100.0, gait_period=50.0):
    """Generate walking joint angles using simple sinusoidal patterns."""
    
    times = np.linspace(0, duration, int(duration * 100))
    N = len(times)
    
    # Initialize
    left_hip = np.zeros(N)
    left_knee = np.zeros(N)
    left_ankle = np.zeros(N)
    right_hip = np.zeros(N)
    right_knee = np.zeros(N)
    right_ankle = np.zeros(N)
    
    standing_knee = -0.15  # Standing position knee angle
    standing_hip = 0.0
    standing_ankle = 0.0
    
    for i, t in enumerate(times):
        phase = (t % gait_period) / gait_period
        step_num = int(t / (gait_period / 2))
        
        # LEFT LEG: swings on odd steps
        if step_num % 2 == 1:
            # Swing phase: hip moves forward, knee flexes then extends
            local_phase = (phase if phase < 0.5 else phase - 0.5) * 2.0
            left_hip[i] = standing_hip - 0.6 * np.sin(local_phase * np.pi)  # Negative = forward swing
            left_knee[i] = standing_knee - 0.3 * (1 - np.cos(local_phase * np.pi))  # Knee flexes in swing
            left_ankle[i] = standing_ankle
        else:
            # Stance phase: hip stays neutral or slightly back
            left_hip[i] = standing_hip - 0.2 * (1 - np.cos(phase * np.pi * 2))  # Slight forward bias
            left_knee[i] = standing_knee  # Keep knee extended in stance
            left_ankle[i] = standing_ankle
        
        # RIGHT LEG: swings on even steps
        if step_num % 2 == 0:
            # Swing phase
            local_phase = (phase if phase < 0.5 else phase - 0.5) * 2.0
            right_hip[i] = standing_hip - 0.6 * np.sin(local_phase * np.pi)  # Negative = forward swing
            right_knee[i] = standing_knee - 0.3 * (1 - np.cos(local_phase * np.pi))  # Knee flexes in swing
            right_ankle[i] = standing_ankle
        else:
            # Stance phase: hip stays neutral or slightly back
            right_hip[i] = standing_hip - 0.2 * (1 - np.cos(phase * np.pi * 2))  # Slight forward bias
            right_knee[i] = standing_knee  # Keep knee extended in stance
            right_ankle[i] = standing_ankle
    
    # Save
    np.save("ik_times.npy", times)
    np.save("ik_left_hip.npy", left_hip)
    np.save("ik_left_knee.npy", left_knee)
    np.save("ik_left_ankle.npy", left_ankle)
    np.save("ik_right_hip.npy", right_hip)
    np.save("ik_right_knee.npy", right_knee)
    np.save("ik_right_ankle.npy", right_ankle)
    
    print("[+] Generated simple walking joint angles")
    print(f"    Left hip range: [{left_hip.min():.4f}, {left_hip.max():.4f}] rad")
    print(f"    Left knee range: [{left_knee.min():.4f}, {left_knee.max():.4f}] rad")
    print(f"    Right hip range: [{right_hip.min():.4f}, {right_hip.max():.4f}] rad")
    print(f"    Right knee range: [{right_knee.min():.4f}, {right_knee.max():.4f}] rad")
    print("[+] Saved: ik_*.npy files")

if __name__ == "__main__":
    generate_simple_walking_angles()
