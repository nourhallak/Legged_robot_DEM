#!/usr/bin/env python3
"""
Walking Trajectory Generator for Legged Robot

Generates smooth, physically-valid bipedal walking trajectories:
- Base (hip) trajectory with forward motion and vertical oscillation
- Foot1 (left) trajectory with alternating stance/swing phases
- Foot2 (right) trajectory offset 180° from foot1

Author: DEM Project
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Trajectory parameters
NUM_STEPS = 400                 # Total trajectory points
STRIDE_LENGTH = 0.005           # Forward progress per step (m)
CYCLE_STEPS = 100               # Steps per half gait cycle
STANCE_FRACTION = 0.60          # 60% stance, 40% swing

# Height parameters
GROUND_Z = 0.210                # Ground contact height (m)
SWING_CLEARANCE = 0.010         # Foot clearance above ground (m)
Z_OSCILLATION_AMP = 0.005       # Hip vertical oscillation amplitude (m)

# Lateral spacing
FOOT_SPACING = 0.020            # Left-right distance between feet (m)

# Derived parameters
Z_MEAN = GROUND_Z + 0.010       # Mean hip height
STANCE_STEPS = int(CYCLE_STEPS * STANCE_FRACTION)
SWING_STEPS = CYCLE_STEPS - STANCE_STEPS

print("="*80)
print("BIPEDAL WALKING TRAJECTORY GENERATOR")
print("="*80)
print(f"\nConfiguration:")
print(f"  Total steps: {NUM_STEPS}")
print(f"  Stride length: {STRIDE_LENGTH*1000:.1f} mm")
print(f"  Gait cycle: {CYCLE_STEPS} steps ({STANCE_FRACTION*100:.0f}% stance, {(1-STANCE_FRACTION)*100:.0f}% swing)")
print(f"  Ground height: {GROUND_Z*1000:.1f} mm")
print(f"  Swing clearance: {SWING_CLEARANCE*1000:.1f} mm")
print(f"  Foot spacing (L-R): {FOOT_SPACING*1000:.1f} mm")
print(f"  Hip Z oscillation: ±{Z_OSCILLATION_AMP*1000:.1f} mm")

# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================

# Initialize trajectory arrays
base_trajectory = np.zeros((NUM_STEPS, 3))
foot1_trajectory = np.zeros((NUM_STEPS, 3))
foot2_trajectory = np.zeros((NUM_STEPS, 3))

# Generate trajectories
for step in range(NUM_STEPS):
    # ========== BASE (HIP) TRAJECTORY ==========
    # X: constant forward velocity
    base_trajectory[step, 0] = STRIDE_LENGTH * step
    
    # Y: centered (walk in straight line)
    base_trajectory[step, 1] = 0.0
    
    # Z: smooth vertical oscillation (inverted pendulum model)
    cycle_phase = (step % CYCLE_STEPS) / CYCLE_STEPS
    z_phase = 2 * np.pi * cycle_phase
    base_trajectory[step, 2] = Z_MEAN + Z_OSCILLATION_AMP * np.cos(z_phase)
    
    # ========== FOOT 1 (LEFT) TRAJECTORY ==========
    cycle_pos = step % CYCLE_STEPS
    
    if cycle_pos < STANCE_STEPS:
        # STANCE PHASE: Foot planted on ground
        # Calculate which contact point we're on
        contact_cycle = step // CYCLE_STEPS
        foot1_trajectory[step, 0] = STRIDE_LENGTH * contact_cycle
        foot1_trajectory[step, 1] = -FOOT_SPACING
        foot1_trajectory[step, 2] = GROUND_Z
    else:
        # SWING PHASE: Foot lifted and moved forward
        swing_progress = (cycle_pos - STANCE_STEPS) / SWING_STEPS  # 0 to 1
        
        # Current and next contact points
        contact_cycle = step // CYCLE_STEPS
        current_contact = STRIDE_LENGTH * contact_cycle
        next_contact = STRIDE_LENGTH * (contact_cycle + 1)
        
        # X: smooth forward arc during swing
        foot1_trajectory[step, 0] = (current_contact + 
                                     (next_contact - current_contact) * swing_progress)
        
        # Y: maintain lateral offset
        foot1_trajectory[step, 1] = -FOOT_SPACING
        
        # Z: parabolic arc (sine wave) - rises and falls
        lift = SWING_CLEARANCE * np.sin(np.pi * swing_progress)
        foot1_trajectory[step, 2] = GROUND_Z + lift
    
    # ========== FOOT 2 (RIGHT) TRAJECTORY ==========
    # Phase offset: 180° out of phase with foot1
    cycle_pos2 = (step + CYCLE_STEPS // 2) % CYCLE_STEPS
    
    if cycle_pos2 < STANCE_STEPS:
        # STANCE PHASE
        contact_cycle2 = (step + CYCLE_STEPS // 2) // CYCLE_STEPS
        foot2_trajectory[step, 0] = STRIDE_LENGTH * contact_cycle2
        foot2_trajectory[step, 1] = FOOT_SPACING
        foot2_trajectory[step, 2] = GROUND_Z
    else:
        # SWING PHASE
        swing_progress2 = (cycle_pos2 - STANCE_STEPS) / SWING_STEPS
        
        contact_cycle2 = (step + CYCLE_STEPS // 2) // CYCLE_STEPS
        current_contact2 = STRIDE_LENGTH * contact_cycle2
        next_contact2 = STRIDE_LENGTH * (contact_cycle2 + 1)
        
        foot2_trajectory[step, 0] = (current_contact2 + 
                                     (next_contact2 - current_contact2) * swing_progress2)
        foot2_trajectory[step, 1] = FOOT_SPACING
        
        lift2 = SWING_CLEARANCE * np.sin(np.pi * swing_progress2)
        foot2_trajectory[step, 2] = GROUND_Z + lift2

# ============================================================================
# VALIDATION & ANALYSIS
# ============================================================================

print(f"\n{'='*80}")
print("TRAJECTORY STATISTICS")
print(f"{'='*80}")

print(f"\nBase (Hip) Trajectory:")
print(f"  X range: {base_trajectory[:, 0].min()*1000:7.2f} to {base_trajectory[:, 0].max()*1000:7.2f} mm")
print(f"  Total forward distance: {(base_trajectory[-1, 0] - base_trajectory[0, 0])*1000:.2f} mm")
print(f"  Y: {base_trajectory[:, 1].min()*1000:7.2f} to {base_trajectory[:, 1].max()*1000:7.2f} mm")
print(f"  Z range: {base_trajectory[:, 2].min()*1000:7.2f} to {base_trajectory[:, 2].max()*1000:7.2f} mm")
print(f"  Z oscillation: ±{(base_trajectory[:, 2].max() - base_trajectory[:, 2].min())/2*1000:.2f} mm")

print(f"\nFoot1 (Left) Trajectory:")
print(f"  X range: {foot1_trajectory[:, 0].min()*1000:7.2f} to {foot1_trajectory[:, 0].max()*1000:7.2f} mm")
print(f"  Y: {foot1_trajectory[:, 1].min()*1000:7.2f} to {foot1_trajectory[:, 1].max()*1000:7.2f} mm")
print(f"  Z range: {foot1_trajectory[:, 2].min()*1000:7.2f} to {foot1_trajectory[:, 2].max()*1000:7.2f} mm")

print(f"\nFoot2 (Right) Trajectory:")
print(f"  X range: {foot2_trajectory[:, 0].min()*1000:7.2f} to {foot2_trajectory[:, 0].max()*1000:7.2f} mm")
print(f"  Y: {foot2_trajectory[:, 1].min()*1000:7.2f} to {foot2_trajectory[:, 1].max()*1000:7.2f} mm")
print(f"  Z range: {foot2_trajectory[:, 2].min()*1000:7.2f} to {foot2_trajectory[:, 2].max()*1000:7.2f} mm")

# Validation checks
print(f"\n{'='*80}")
print("VALIDATION CHECKS")
print(f"{'='*80}")

checks_passed = 0
checks_total = 0

# Check 1: Feet don't go below ground
checks_total += 1
if foot1_trajectory[:, 2].min() >= GROUND_Z:
    print(f"✓ Foot1 stays above ground (min: {foot1_trajectory[:, 2].min()*1000:.2f} mm)")
    checks_passed += 1
else:
    print(f"✗ Foot1 goes below ground!")

checks_total += 1
if foot2_trajectory[:, 2].min() >= GROUND_Z:
    print(f"✓ Foot2 stays above ground (min: {foot2_trajectory[:, 2].min()*1000:.2f} mm)")
    checks_passed += 1
else:
    print(f"✗ Foot2 goes below ground!")

# Check 2: Feet reach maximum height
checks_total += 1
max_foot_z = max(foot1_trajectory[:, 2].max(), foot2_trajectory[:, 2].max())
if max_foot_z <= GROUND_Z + SWING_CLEARANCE + 0.001:  # Small tolerance
    print(f"✓ Foot swing heights within limits (max: {max_foot_z*1000:.2f} mm)")
    checks_passed += 1
else:
    print(f"✗ Foot swing height exceeds limit!")

# Check 3: Forward motion is monotonic
checks_total += 1
if np.all(np.diff(base_trajectory[:, 0]) >= 0):
    print(f"✓ Forward motion is monotonic (no backward stepping)")
    checks_passed += 1
else:
    print(f"✗ Forward motion has reversals!")

# Check 4: Walking in straight line
checks_total += 1
if abs(base_trajectory[:, 1].max() - base_trajectory[:, 1].min()) < 0.001:
    print(f"✓ Walking in straight line (Y deviation < 1mm)")
    checks_passed += 1
else:
    print(f"✗ Significant lateral deviation!")

# Check 5: Feet alternate (no simultaneous flight)
checks_total += 1
foot1_in_air = foot1_trajectory[:, 2] > (GROUND_Z + 0.001)
foot2_in_air = foot2_trajectory[:, 2] > (GROUND_Z + 0.001)
simultaneous_flight = np.sum(foot1_in_air & foot2_in_air)
if simultaneous_flight == 0:
    print(f"✓ Feet alternate properly (no simultaneous flight)")
    checks_passed += 1
else:
    print(f"✗ Feet simultaneously airborne for {simultaneous_flight} steps!")

print(f"\n{'='*80}")
print(f"Validation: {checks_passed}/{checks_total} checks passed")
print(f"{'='*80}")

# ============================================================================
# SAVE TRAJECTORIES
# ============================================================================

print(f"\nSaving trajectories...")

# Create output directory if needed
output_dir = Path(".")

# Save numpy arrays
np.save(output_dir / "base_trajectory.npy", base_trajectory)
np.save(output_dir / "foot1_trajectory.npy", foot1_trajectory)
np.save(output_dir / "foot2_trajectory.npy", foot2_trajectory)

print(f"✓ base_trajectory.npy   (shape: {base_trajectory.shape})")
print(f"✓ foot1_trajectory.npy  (shape: {foot1_trajectory.shape})")
print(f"✓ foot2_trajectory.npy  (shape: {foot2_trajectory.shape})")

# ============================================================================
# VISUALIZATION
# ============================================================================

print(f"\nGenerating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Bipedal Walking Trajectories", fontsize=16, fontweight='bold')

# X positions
ax = axes[0, 0]
ax.plot(base_trajectory[:, 0]*1000, label='Base (Hip)', linewidth=2)
ax.plot(foot1_trajectory[:, 0]*1000, label='Foot1 (Left)', linewidth=1.5, alpha=0.7)
ax.plot(foot2_trajectory[:, 0]*1000, label='Foot2 (Right)', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Step')
ax.set_ylabel('X Position (mm)')
ax.set_title('Forward Motion')
ax.legend()
ax.grid(True, alpha=0.3)

# Y positions
ax = axes[0, 1]
ax.plot(base_trajectory[:, 1]*1000, label='Base (Hip)', linewidth=2)
ax.plot(foot1_trajectory[:, 1]*1000, label='Foot1 (Left)', linewidth=1.5, alpha=0.7)
ax.plot(foot2_trajectory[:, 1]*1000, label='Foot2 (Right)', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Step')
ax.set_ylabel('Y Position (mm)')
ax.set_title('Lateral Motion')
ax.legend()
ax.grid(True, alpha=0.3)

# Z positions
ax = axes[0, 2]
ax.plot(base_trajectory[:, 2]*1000, label='Base (Hip)', linewidth=2)
ax.plot(foot1_trajectory[:, 2]*1000, label='Foot1 (Left)', linewidth=1.5, alpha=0.7)
ax.plot(foot2_trajectory[:, 2]*1000, label='Foot2 (Right)', linewidth=1.5, alpha=0.7)
ax.axhline(GROUND_Z*1000, color='brown', linestyle='--', label='Ground', linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('Z Position (mm)')
ax.set_title('Vertical Motion')
ax.legend()
ax.grid(True, alpha=0.3)

# 3D view (XZ plane)
ax = axes[1, 0]
ax.plot(base_trajectory[:, 0]*1000, base_trajectory[:, 2]*1000, label='Base', linewidth=2)
ax.plot(foot1_trajectory[:, 0]*1000, foot1_trajectory[:, 2]*1000, label='Foot1', linewidth=1.5, alpha=0.7)
ax.plot(foot2_trajectory[:, 0]*1000, foot2_trajectory[:, 2]*1000, label='Foot2', linewidth=1.5, alpha=0.7)
ax.axhline(GROUND_Z*1000, color='brown', linestyle='--', linewidth=2)
ax.set_xlabel('X Position (mm)')
ax.set_ylabel('Z Position (mm)')
ax.set_title('Side View (XZ Plane)')
ax.legend()
ax.grid(True, alpha=0.3)

# 3D view (YZ plane)
ax = axes[1, 1]
ax.plot(base_trajectory[:, 1]*1000, base_trajectory[:, 2]*1000, label='Base', linewidth=2)
ax.plot(foot1_trajectory[:, 1]*1000, foot1_trajectory[:, 2]*1000, label='Foot1', linewidth=1.5, alpha=0.7)
ax.plot(foot2_trajectory[:, 1]*1000, foot2_trajectory[:, 2]*1000, label='Foot2', linewidth=1.5, alpha=0.7)
ax.axhline(GROUND_Z*1000, color='brown', linestyle='--', linewidth=2)
ax.set_xlabel('Y Position (mm)')
ax.set_ylabel('Z Position (mm)')
ax.set_title('Front View (YZ Plane)')
ax.legend()
ax.grid(True, alpha=0.3)

# Phase diagram
ax = axes[1, 2]
steps_range = np.arange(min(150, NUM_STEPS))  # First 150 steps for clarity
foot1_in_air = foot1_trajectory[steps_range, 2] > (GROUND_Z + 0.001)
foot2_in_air = foot2_trajectory[steps_range, 2] > (GROUND_Z + 0.001)

ax.fill_between(steps_range, 0, foot1_in_air*1, alpha=0.5, label='Foot1 Swing', color='blue')
ax.fill_between(steps_range, 1, 1 + foot2_in_air*1, alpha=0.5, label='Foot2 Swing', color='red')
ax.set_ylim(-0.1, 2.1)
ax.set_xlim(0, len(steps_range))
ax.set_xlabel('Step')
ax.set_ylabel('Phase')
ax.set_title('Gait Phase (First 150 steps)')
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(['Foot1', 'Foot2'])
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig("walking_trajectories.png", dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: walking_trajectories.png")

print(f"\n{'='*80}")
print("TRAJECTORY GENERATION COMPLETE")
print(f"{'='*80}")
print(f"\nNext steps:")
print(f"1. Run IK solver to find joint angles for each trajectory point")
print(f"2. Validate IK accuracy on generated trajectories")
print(f"3. Run MuJoCo simulation with PD joint controllers")
print(f"4. Monitor for sliding, jumping, or instability issues")
print(f"\nTrajectory parameters can be customized by editing the configuration section")
print(f"at the top of this script.")
