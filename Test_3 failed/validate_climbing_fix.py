"""
Validate that the reduced velocity (0.0015 m/s) fixes climbing issue.
Re-runs the IK solver and compares foot heights across all 10 steps.
"""

import numpy as np
import mujoco
import os

# Load the model
model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

# Load trajectories
traj_data = np.load('biped_10step_sinusoidal.npy', allow_pickle=True).item()
ik_data = np.load('biped_ik_solutions.npy', allow_pickle=True).item()

left_trajectory = traj_data['left_trajectory']
right_trajectory = traj_data['right_trajectory']
left_ik = ik_data['left_joint_angles']
right_ik = ik_data['right_joint_angles']

print("=" * 70)
print("CLIMBING FIX VALIDATION - Reduced Velocity Test")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Forward velocity: 0.0015 m/s (reduced from 0.003)")
print(f"  Z velocity bias: -0.0001 m/s (reduced from -0.0003)")
print(f"\nTesting foot Z-heights across all 10 steps...")
print("\n" + "=" * 70)

# Test parameters - adjusted for 500-point trajectory (not 1000)
step_frames = [37, 87, 137, 187, 237, 287, 337, 387, 437, 487]  # Every 50 points = 10 steps
step_numbers = list(range(1, 11))

# Store results
results = []

for step_num, frame_idx in zip(step_numbers, step_frames):
    # Set joint angles from IK solution
    data.qpos[3:6] = left_ik[frame_idx]  # Left leg
    data.qpos[6:9] = right_ik[frame_idx]  # Right leg
    
    # Compute forward kinematics
    mujoco.mj_forward(model, data)
    
    # Get foot positions (from body positions)
    # Body 4 is left foot (foot_1), Body 7 is right foot (foot_2)
    left_foot_pos = data.xpos[4]  # Body 4 is left foot
    right_foot_pos = data.xpos[7]  # Body 7 is right foot
    
    # Target positions from trajectory
    left_target = left_trajectory[frame_idx]
    right_target = right_trajectory[frame_idx]
    
    # Compute errors
    left_error = np.linalg.norm(left_foot_pos - left_target)
    right_error = np.linalg.norm(right_foot_pos - right_target)
    
    # Get Z coordinates
    left_z = left_foot_pos[2]
    right_z = right_foot_pos[2]
    target_z = left_target[2]
    
    # Ground contact gap (should be ~0 at floor Z=0.431)
    floor_z = 0.431
    left_gap = left_z - floor_z
    right_gap = right_z - floor_z
    
    results.append({
        'step': step_num,
        'frame': frame_idx,
        'left_z': left_z,
        'right_z': right_z,
        'target_z': target_z,
        'left_gap': left_gap,
        'right_gap': right_gap,
        'left_error': left_error,
        'right_error': right_error,
    })
    
    # Determine if climbing (Z > 1mm above target)
    climbing = (left_z > target_z + 0.001) or (right_z > target_z + 0.001)
    status = "⚠️ CLIMBING" if climbing else "✅ OK"
    
    print(f"\nStep {step_num} (Frame {frame_idx}):")
    print(f"  Target Z: {target_z*1000:7.2f}mm")
    print(f"  Left foot:  Z={left_z*1000:7.2f}mm | Gap={left_gap*1000:6.2f}mm | Error={left_error*1000:5.2f}mm")
    print(f"  Right foot: Z={right_z*1000:7.2f}mm | Gap={right_gap*1000:6.2f}mm | Error={right_error*1000:5.2f}mm")
    print(f"  Status: {status}")

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

z_heights = [r['left_z'] for r in results] + [r['right_z'] for r in results]
target_heights = [r['target_z'] for r in results] * 2
gaps = [r['left_gap'] for r in results] + [r['right_gap'] for r in results]

print(f"\nZ-Height Analysis:")
print(f"  Min Z: {min(z_heights)*1000:.2f}mm")
print(f"  Max Z: {max(z_heights)*1000:.2f}mm")
print(f"  Range: {(max(z_heights) - min(z_heights))*1000:.2f}mm")
print(f"  Target: {results[0]['target_z']*1000:.2f}mm")

print(f"\nGround Contact Gaps:")
print(f"  Min gap: {min(gaps)*1000:.2f}mm")
print(f"  Max gap: {max(gaps)*1000:.2f}mm")
print(f"  Mean gap: {np.mean(gaps)*1000:.2f}mm")

# Check if climbing is fixed
max_drift = max(z_heights) - results[0]['target_z']
climbing_fixed = max_drift < 0.003  # < 3mm drift

print(f"\nClimbing Status:")
if climbing_fixed:
    print(f"  ✅ CLIMBING FIXED! Max drift: {max_drift*1000:.2f}mm (< 3mm threshold)")
else:
    print(f"  ❌ CLIMBING PERSISTS! Max drift: {max_drift*1000:.2f}mm (>= 3mm threshold)")

# Check progression
print(f"\nProgression Analysis:")
for i, result in enumerate(results):
    avg_z = (result['left_z'] + result['right_z']) / 2
    drift = (avg_z - results[0]['target_z']) * 1000
    print(f"  Step {result['step']:2d}: {drift:6.2f}mm drift from step 1")

print("\n" + "=" * 70)
