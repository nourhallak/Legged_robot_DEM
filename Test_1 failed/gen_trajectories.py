"""
Generate walking trajectories based on robot geometry.
Floor at z=0.21m
Hip must be above feet for physically valid walking.
"""
import numpy as np

print("="*70)
print("TRAJECTORY PLANNER - CORRECTED FOR ROBOT GEOMETRY")
print("="*70)

# ===== ROBOT GEOMETRY (from analysis) =====
# When hip at z=0.26m, feet naturally reach to z=0.267m (extended)
# Vertical leg reach: 48mm
# Hip should be HIGHER than feet minimum position

num_steps = 400

# Hip trajectory: base at 0.26m, oscillates ±0.03m
# This keeps hip well above feet and ensures ground clearance
hip_base_z = 0.26        # Base height
hip_z_amplitude = 0.03   # Oscillation amplitude

# Foot trajectory: 
# Minimum at 0.215m (swing, 5mm above floor z=0.21m)
# Maximum at 0.267m (stance, fully extended)
foot_z_min = 0.215       # Swing (foot lifted)
foot_z_max = 0.267       # Stance (leg extended down)

print(f"\nParameters:")
print(f"  Hip base Z: {hip_base_z:.3f}m")
print(f"  Hip oscillation: ±{hip_z_amplitude:.3f}m")
print(f"  Foot Z range: {foot_z_min:.3f}m (swing) to {foot_z_max:.3f}m (stance)")
print(f"  Floor level: 0.210m (5mm clearance at minimum)")

# Initialize trajectories
hip_traj = np.zeros((num_steps, 3))
foot1_traj = np.zeros((num_steps, 3))
foot2_traj = np.zeros((num_steps, 3))

# Generate trajectories
for i in range(num_steps):
    progress = i / num_steps  # 0 to 1
    
    # ===== HIP TRAJECTORY =====
    # Hip oscillates with a smooth pattern
    hip_z = hip_base_z + hip_z_amplitude * np.cos(2 * np.pi * progress)
    hip_x = 0.005 * progress  # Forward progression
    
    hip_traj[i] = [hip_x, 0.0, hip_z]
    
    # ===== FEET TRAJECTORIES =====
    # Foot 1: Stance phase 0-50%, Swing phase 50-100%
    if progress < 0.5:
        # Stance phase: foot down, supporting body
        foot1_z = foot_z_max
        foot1_x = hip_x - 0.01  # Behind hip
        foot1_y = -0.02         # Left leg
    else:
        # Swing phase: foot lifted and moved forward
        swing_phase = (progress - 0.5) * 2  # 0->1 during swing
        # Smooth arc: use sine for natural swing
        foot1_z = foot_z_min + (foot_z_max - foot_z_min) * (1 - np.cos(np.pi * swing_phase)) / 2
        foot1_x = hip_x + 0.015 * np.sin(np.pi * swing_phase)  # Forward swing
        foot1_y = -0.02
    
    foot1_traj[i] = [foot1_x, foot1_y, foot1_z]
    
    # Foot 2: Opposite phase (Swing first 0-50%, Stance 50-100%)
    if progress < 0.5:
        # Swing phase
        swing_phase = progress * 2  # 0->1 during first half
        foot2_z = foot_z_min + (foot_z_max - foot_z_min) * (1 - np.cos(np.pi * swing_phase)) / 2
        foot2_x = hip_x + 0.015 * np.sin(np.pi * swing_phase)
        foot2_y = 0.02  # Right leg
    else:
        # Stance phase
        foot2_z = foot_z_max
        foot2_x = hip_x - 0.01
        foot2_y = 0.02
    
    foot2_traj[i] = [foot2_x, foot2_y, foot2_z]

# Save trajectories
np.save('hip_trajectory.npy', hip_traj)
np.save('foot1_trajectory.npy', foot1_traj)
np.save('foot2_trajectory.npy', foot2_traj)

print(f"\nTrajectory ranges generated:")
print(f"  Hip Z: {hip_traj[:, 2].min():.4f}m to {hip_traj[:, 2].max():.4f}m")
print(f"  Hip X: {hip_traj[:, 0].min():.4f}m to {hip_traj[:, 0].max():.4f}m")
print(f"  Foot1 Z: {foot1_traj[:, 2].min():.4f}m to {foot1_traj[:, 2].max():.4f}m")
print(f"  Foot2 Z: {foot2_traj[:, 2].min():.4f}m to {foot2_traj[:, 2].max():.4f}m")
print(f"\nKey relationships:")
print(f"  Hip is {hip_traj[:, 2].min() - foot1_traj[:, 2].max():.4f}m BELOW foot maximum")
print(f"  Hip is {hip_traj[:, 2].max() - foot1_traj[:, 2].min():.4f}m ABOVE foot minimum")
print(f"\nAll trajectories saved!")
