#!/usr/bin/env python3
"""Generate walking trajectories within the robot's actual workspace."""

import numpy as np

def generate_realistic_gait(duration=100.0, gait_period=50.0):
    """
    Generate walking trajectories that fit the robot's actual workspace.
    
    Robot's actual reachable X: -0.0215 to +0.0223 m (4.4 cm total)
    So we use small steps: ~1.3 cm per step
    """
    
    # Time array
    times = np.linspace(0, duration, int(duration * 100))
    num_points = len(times)
    
    # Initialize trajectories
    left_foot = np.zeros((num_points, 3))
    right_foot = np.zeros((num_points, 3))
    left_angles = np.zeros((num_points, 3))
    right_angles = np.zeros((num_points, 3))
    base_pos = np.zeros((num_points, 3))
    
    # Real workspace parameters
    step_length = 0.01  # 1 cm per step (within 4.4 cm workspace)
    step_height = 0.015  # 1.5 cm swing height
    
    # Standing pose
    stand_x = 0.005  # Middle of reachable range
    stand_z = 0.485  # Ground contact height
    
    print(f"Generating realistic walking trajectories...")
    print(f"  Step length: {step_length*100:.1f} cm")
    print(f"  Step height: {step_height*100:.1f} cm")
    print(f"  Gait period: {gait_period:.1f} s")
    
    for i, t in enumerate(times):
        # Phase in gait cycle (0 to 1)
        phase = (t % gait_period) / gait_period
        
        # How many complete steps
        step_number = int(t / (gait_period / 2.0))
        
        # Base forward motion (slow and steady)
        base_x = (step_length / 2) * (t / gait_period) * 5  # Advance slowly
        
        # LEFT FOOT - moves on odd steps (1, 3, 5, ...)
        if step_number % 2 == 1:  # Left foot is swinging
            # Local phase (0 to 1 during swing)
            local_phase = (phase if phase < 0.5 else phase - 0.5) * 2.0
            # Swings forward from back leg position
            left_foot[i, 0] = base_x + step_length * local_phase
            left_foot[i, 2] = stand_z + step_height * np.sin(local_phase * np.pi)
        else:  # Left foot is in stance
            left_foot[i, 0] = base_x + step_length
            left_foot[i, 2] = stand_z
        
        # RIGHT FOOT - moves on even steps (0, 2, 4, ...)
        if step_number % 2 == 0:  # Right foot is swinging
            local_phase = (phase if phase < 0.5 else phase - 0.5) * 2.0
            right_foot[i, 0] = base_x + step_length * local_phase
            right_foot[i, 2] = stand_z + step_height * np.sin(local_phase * np.pi)
        else:  # Right foot is in stance
            right_foot[i, 0] = base_x + step_length
            right_foot[i, 2] = stand_z
        
        # Foot Y (lateral) - minimal motion, centered between legs
        left_foot[i, 1] = -0.0070  # Left side
        right_foot[i, 1] = -0.0064  # Right side
        
        # Joint angles - coordinate with foot position
        # Hip extends during swing, contracts during stance
        if phase < 0.25:
            # First quarter: right leg swinging, left extending
            phase_swing = phase * 4  # 0 to 1
            left_angles[i, 0] = 0.4 + 0.2 * phase_swing  # Hip extending
            left_angles[i, 1] = -0.25 - 0.05 * phase_swing  # Knee extending
            left_angles[i, 2] = 0.0  # Ankle neutral
            
            right_angles[i, 0] = 0.2 * phase_swing  # Hip swinging forward
            right_angles[i, 1] = -0.15 + 0.1 * phase_swing  # Knee flexing
            right_angles[i, 2] = 0.0
        elif phase < 0.5:
            # Second quarter: transition
            phase_swing = (phase - 0.25) * 4
            left_angles[i, 0] = 0.6 - 0.2 * phase_swing
            left_angles[i, 1] = -0.30 - 0.05 * (1 - phase_swing)
            left_angles[i, 2] = 0.0
            
            right_angles[i, 0] = 0.2 + 0.2 * phase_swing
            right_angles[i, 1] = -0.05 + 0.1 * (1 - phase_swing)
            right_angles[i, 2] = 0.0
        elif phase < 0.75:
            # Third quarter: left leg swinging, right extending
            phase_swing = (phase - 0.5) * 4
            right_angles[i, 0] = 0.4 + 0.2 * phase_swing
            right_angles[i, 1] = -0.25 - 0.05 * phase_swing
            right_angles[i, 2] = 0.0
            
            left_angles[i, 0] = 0.2 * phase_swing
            left_angles[i, 1] = -0.15 + 0.1 * phase_swing
            left_angles[i, 2] = 0.0
        else:
            # Fourth quarter: transition
            phase_swing = (phase - 0.75) * 4
            right_angles[i, 0] = 0.6 - 0.2 * phase_swing
            right_angles[i, 1] = -0.30 - 0.05 * (1 - phase_swing)
            right_angles[i, 2] = 0.0
            
            left_angles[i, 0] = 0.2 + 0.2 * phase_swing
            left_angles[i, 1] = -0.05 + 0.1 * (1 - phase_swing)
            left_angles[i, 2] = 0.0
        
        # Base position - between feet
        base_pos[i, 0] = base_x + step_length / 2
        base_pos[i, 1] = -0.0067  # Midpoint between feet laterally
        base_pos[i, 2] = 0.475  # Stable height
    
    # Save trajectories
    np.save('traj_times.npy', times)
    np.save('traj_left_foot.npy', left_foot)
    np.save('traj_right_foot.npy', right_foot)
    np.save('traj_left_angles.npy', left_angles)
    np.save('traj_right_angles.npy', right_angles)
    np.save('traj_base_pos.npy', base_pos)
    
    # Summary
    print(f"\n[TRAJECTORY SUMMARY]")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Points: {num_points}")
    print(f"  Left foot X range: [{left_foot[:, 0].min():.5f}, {left_foot[:, 0].max():.5f}] m")
    print(f"  Right foot X range: [{right_foot[:, 0].min():.5f}, {right_foot[:, 0].max():.5f}] m")
    print(f"  Left foot Z range: [{left_foot[:, 2].min():.5f}, {left_foot[:, 2].max():.5f}] m")
    print(f"  Right foot Z range: [{right_foot[:, 2].min():.5f}, {right_foot[:, 2].max():.5f}] m")
    
    # Verify monotonic forward motion
    left_x_diff = np.diff(left_foot[:, 0])
    right_x_diff = np.diff(right_foot[:, 0])
    
    left_forward = np.sum(left_x_diff > -1e-6) / len(left_x_diff) * 100
    right_forward = np.sum(right_x_diff > -1e-6) / len(right_x_diff) * 100
    
    print(f"\n[FORWARD MOTION CHECK]")
    print(f"  Left foot forward {left_forward:.1f}% of the time")
    print(f"  Right foot forward {right_forward:.1f}% of the time")
    
    print(f"\n[OK] Realistic trajectories generated!")

if __name__ == '__main__':
    generate_realistic_gait(duration=100.0, gait_period=50.0)
