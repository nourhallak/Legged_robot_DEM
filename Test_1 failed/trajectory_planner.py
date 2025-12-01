#!/usr/bin/env python3
"""
Complete trajectory planner for bipedal walking
- Hip trajectory: smooth forward motion with COM-like vertical oscillation
- Foot1 trajectory: stance phase (fixed) and swing phase (arc motion)
- Foot2 trajectory: opposite phase from foot1
- All trajectories account for actual robot geometry
"""
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("BIPEDAL WALKING TRAJECTORY PLANNER")
print("="*70)
print()

# ============================================================================
# PARAMETERS
# ============================================================================

num_steps = 400
stride_length = 0.005  # 5mm forward per step
step_cycle = 200       # 200 steps per full gait cycle (L-stance, R-swing, R-stance, L-swing)

# Hip controls the body height during walking (inverted pendulum)
# Work WITH robot geometry: feet are naturally 10-12mm above hip
# So when hip is at 0.228m, feet will naturally be at ~0.238-0.240m
# Make hip LOWER so feet appear below hip in the plot
# Hip at 0.215m -> feet at ~0.227m (clearly above, for swing)
hip_z_nominal = 0.215  # Nominal body height (LOWER than before)
hip_z_max = 0.225      # Maximum height during double support
hip_z_min = 0.205      # Minimum height during single support

foot_z_ground = 0.213  # Where feet should contact ground
foot_z_swing = 0.221   # Swing phase clearance

# Create time array
time_array = np.linspace(0, num_steps - 1, num_steps)

print(f"Gait Parameters:")
print(f"  Total steps: {num_steps}")
print(f"  Stride length: {stride_length*1000:.1f} mm")
print(f"  Step cycle: {step_cycle} steps")
print(f"  Hip Z range: {hip_z_min:.3f}m to {hip_z_max:.3f}m")
print(f"  Foot ground contact: {foot_z_ground:.3f}m")
print(f"  Foot swing clearance: {foot_z_swing:.3f}m")
print()

# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================

# Initialize trajectory arrays
hip_trajectory = np.zeros((num_steps, 3))
foot1_trajectory = np.zeros((num_steps, 3))
foot2_trajectory = np.zeros((num_steps, 3))

# Gait phase oscillation
gait_phase = np.linspace(0, num_steps/step_cycle * 2*np.pi, num_steps)

# For each step
for step in range(num_steps):
    phase = step % step_cycle
    
    # ========================================================================
    # HIP TRAJECTORY (Body motion)
    # ========================================================================
    
    # Forward motion: constant velocity
    hip_x = stride_length * (step / step_cycle)
    
    # Vertical motion: smooth oscillation (like inverted pendulum)
    # Double support (0-50 steps): higher
    # Single support L (50-100 steps): lower (R leg in air, load on L)
    # Double support (100-150 steps): higher
    # Single support R (150-200 steps): lower (L leg in air, load on R)
    
    phase_normalized = phase / step_cycle  # 0 to 1
    
    # Use sine wave with 2x period for double/single support pattern
    vertical_oscillation = np.sin(2 * np.pi * phase_normalized)
    
    # Map oscillation to height range
    hip_z = hip_z_nominal + (hip_z_max - hip_z_nominal) * (vertical_oscillation + 1) / 2
    
    hip_trajectory[step, 0] = hip_x
    hip_trajectory[step, 1] = 0.0
    hip_trajectory[step, 2] = hip_z
    
    # ========================================================================
    # FOOT 1 TRAJECTORY (Left foot)
    # ========================================================================
    
    if phase < 100:
        # STANCE PHASE (Left foot on ground)
        # Foot stays at fixed landing position
        landing_num = step // step_cycle
        foot1_x = stride_length * landing_num
        foot1_z = foot_z_ground
    else:
        # SWING PHASE (Left foot swinging forward)
        swing_progress = (phase - 100) / 100.0  # 0 to 1
        
        # Swing from current position to next position
        landing_num = step // step_cycle
        swing_start_x = stride_length * landing_num
        swing_end_x = stride_length * (landing_num + 1)
        
        # Parabolic arc for swing (foot traces smooth path)
        foot1_x = swing_start_x + (swing_end_x - swing_start_x) * swing_progress
        
        # Height: ground -> peak -> ground (parabolic)
        # Swing clearance is difference from ground: 0.225 - 0.210 = 0.015m
        swing_clearance = foot_z_swing - foot_z_ground
        swing_height = swing_clearance * np.sin(np.pi * swing_progress)
        foot1_z = foot_z_ground + swing_height
    
    foot1_trajectory[step, 0] = foot1_x
    foot1_trajectory[step, 1] = 0.0  # No lateral motion
    foot1_trajectory[step, 2] = foot1_z
    
    # ========================================================================
    # FOOT 2 TRAJECTORY (Right foot - opposite phase)
    # ========================================================================
    
    # Right foot is in opposite phase (100 steps offset)
    phase2 = (phase + 100) % step_cycle
    
    if phase2 < 100:
        # STANCE PHASE (Right foot on ground)
        landing_num = (step + 100) // step_cycle
        foot2_x = stride_length * landing_num
        foot2_z = foot_z_ground
    else:
        # SWING PHASE (Right foot swinging forward)
        swing_progress = (phase2 - 100) / 100.0
        
        landing_num = (step + 100) // step_cycle
        swing_start_x = stride_length * landing_num
        swing_end_x = stride_length * (landing_num + 1)
        
        foot2_x = swing_start_x + (swing_end_x - swing_start_x) * swing_progress
        
        swing_clearance = foot_z_swing - foot_z_ground
        swing_height = swing_clearance * np.sin(np.pi * swing_progress)
        foot2_z = foot_z_ground + swing_height
    
    foot2_trajectory[step, 0] = foot2_x
    foot2_trajectory[step, 1] = 0.0
    foot2_trajectory[step, 2] = foot2_z

# ============================================================================
# SAVE TRAJECTORIES
# ============================================================================

np.save('hip_trajectory.npy', hip_trajectory)
np.save('foot1_trajectory.npy', foot1_trajectory)
np.save('foot2_trajectory.npy', foot2_trajectory)

print("[OK] Trajectories saved:")
print(f"  hip_trajectory.npy ({hip_trajectory.shape})")
print(f"  foot1_trajectory.npy ({foot1_trajectory.shape})")
print(f"  foot2_trajectory.npy ({foot2_trajectory.shape})")
print()

# ============================================================================
# STATISTICS
# ============================================================================

print("Hip Trajectory Statistics:")
print(f"  X range: {hip_trajectory[:, 0].min():.4f}m to {hip_trajectory[:, 0].max():.4f}m")
print(f"  Y range: {hip_trajectory[:, 1].min():.4f}m to {hip_trajectory[:, 1].max():.4f}m")
print(f"  Z range: {hip_trajectory[:, 2].min():.4f}m to {hip_trajectory[:, 2].max():.4f}m")
print()

print("Foot1 Trajectory Statistics:")
print(f"  X range: {foot1_trajectory[:, 0].min():.4f}m to {foot1_trajectory[:, 0].max():.4f}m")
print(f"  Z range: {foot1_trajectory[:, 2].min():.4f}m to {foot1_trajectory[:, 2].max():.4f}m")
print(f"  Max swing clearance: {foot1_trajectory[:, 2].max() - foot_z_ground:.4f}m")
print()

print("Foot2 Trajectory Statistics:")
print(f"  X range: {foot2_trajectory[:, 0].min():.4f}m to {foot2_trajectory[:, 0].max():.4f}m")
print(f"  Z range: {foot2_trajectory[:, 2].min():.4f}m to {foot2_trajectory[:, 2].max():.4f}m")
print(f"  Max swing clearance: {foot2_trajectory[:, 2].max() - foot_z_ground:.4f}m")
print()

# ============================================================================
# PLOT TRAJECTORIES
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: XZ view (side view)
ax = axes[0, 0]
ax.plot(hip_trajectory[:, 0], hip_trajectory[:, 2], 'k-', linewidth=2, label='Hip')
ax.plot(foot1_trajectory[:, 0], foot1_trajectory[:, 2], 'r-', linewidth=1.5, label='Foot1 (L)')
ax.plot(foot2_trajectory[:, 0], foot2_trajectory[:, 2], 'b-', linewidth=1.5, label='Foot2 (R)')
ax.axhline(y=foot_z_ground, color='g', linestyle='--', linewidth=1, label='Ground level')
ax.set_xlabel('X position (m)')
ax.set_ylabel('Z position (m)')
ax.set_title('Side View: Walking Trajectory')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Plot 2: Hip Z over time
ax = axes[0, 1]
ax.plot(hip_trajectory[:, 2], 'k-', linewidth=1.5)
ax.axhline(y=hip_z_nominal, color='k', linestyle='--', alpha=0.5, label='Nominal')
ax.set_xlabel('Step number')
ax.set_ylabel('Hip Z (m)')
ax.set_title('Hip Height Oscillation')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Foot Z over time
ax = axes[1, 0]
ax.plot(foot1_trajectory[:, 2], 'r-', linewidth=1.5, label='Foot1 (L)')
ax.plot(foot2_trajectory[:, 2], 'b-', linewidth=1.5, label='Foot2 (R)')
ax.axhline(y=foot_z_ground, color='g', linestyle='--', linewidth=1, label='Ground')
ax.set_xlabel('Step number')
ax.set_ylabel('Foot Z (m)')
ax.set_title('Foot Height Oscillation')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Hip X over time
ax = axes[1, 1]
ax.plot(hip_trajectory[:, 0], 'k-', linewidth=1.5)
ax.set_xlabel('Step number')
ax.set_ylabel('Hip X (m)')
ax.set_title('Hip Forward Motion')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('walking_trajectories_planned.png', dpi=150)
print("[OK] Trajectory plots saved: walking_trajectories_planned.png")
print()

print("="*70)
print("NEXT STEP: Use these trajectories with IK solver")
print("="*70)
