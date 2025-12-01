"""
Generate realistic walking trajectories with proper gait dynamics
- Alternating foot stance/swing phases
- COM vertical oscillation
- Smooth forward progression
- ADJUSTED for kinematic constraints: feet limited to Z=0.207-0.262m
"""
import numpy as np

print("Generating realistic walking trajectories...\n")

# ===== KINEMATIC CONSTRAINTS FROM ROBOT GEOMETRY =====
# From forward kinematics analysis:
# - Minimum Z (stance phase): 0.2069 m
# - Maximum Z (swing phase): 0.2623 m
# - Available swing clearance: ~0.055 m
FOOT_Z_STANCE = 0.207    # Lowest reachable Z for stance phase
FOOT_Z_SWING = 0.257     # Z height for swing phase (0.05m clearance)
COM_Z_MIN = 0.025        # Relative minimum COM height (0.232m absolute = 0.207 + 0.025)
COM_Z_MAX = 0.055        # Relative maximum COM height (0.262m absolute = 0.207 + 0.055)

# Walking parameters
num_steps = 400
stride_length = 0.16  # Total forward distance
com_height_offset = 0.038  # COM height offset above foot stance level
com_bounce = 0.015   # COM vertical oscillation amplitude

# Create time vector
t = np.linspace(0, 1, num_steps)

# --- COM TRAJECTORY ---
# Oscillates vertically as weight shifts between legs
# COM Z: varies between (FOOT_Z_STANCE + COM_Z_MIN) and (FOOT_Z_SWING + COM_Z_MAX)
com_x = stride_length * t  # Linear forward progression
com_y = 0.005 * np.sin(4 * np.pi * t)  # Lateral sway
com_z_base = FOOT_Z_STANCE + com_height_offset  # Nominal COM height
com_z = com_z_base + com_bounce * np.cos(4 * np.pi * t)  # Vertical oscillation

com_trajectory = np.column_stack([com_x, com_y, com_z])

# --- FOOT 1 TRAJECTORY (Left leg) ---
# Foot 1 has swing phase in first half, stance in second half
foot1_phase = (t * 2) % 2  # 0-1: swing, 1-2: stance (wraps back to 0)

# Initialize foot positions (all x, y, z)
foot1_x = np.zeros(num_steps)
foot1_y = -0.01 * np.ones(num_steps)  # Left side
foot1_z = FOOT_Z_STANCE * np.ones(num_steps)  # Start at stance height

# Swing phase: lifted and moving forward
for i in range(num_steps):
    phase = foot1_phase[i]
    if phase <= 1.0:  # Swing phase (first half)
        swing_fraction = phase
        foot1_x[i] = (stride_length / 2) * swing_fraction + stride_length / 2 * (t[i] // 0.5)
        # Foot swing: parabolic trajectory from FOOT_Z_STANCE to FOOT_Z_SWING and back
        swing_lift = (FOOT_Z_SWING - FOOT_Z_STANCE) * np.sin(np.pi * swing_fraction)
        foot1_z[i] = FOOT_Z_STANCE + swing_lift
    else:  # Stance phase (second half)
        foot1_x[i] = stride_length / 2 + stride_length / 2 * (t[i] // 0.5)
        foot1_z[i] = FOOT_Z_STANCE

foot1_trajectory = np.column_stack([foot1_x, foot1_y, foot1_z])

# --- FOOT 2 TRAJECTORY (Right leg) ---
# Foot 2 has opposite phase (swing when foot1 is stance)
foot2_phase = ((t * 2 + 1) % 2)  # Phase-shifted by 1

foot2_x = np.zeros(num_steps)
foot2_y = 0.01 * np.ones(num_steps)  # Right side
foot2_z = FOOT_Z_STANCE * np.ones(num_steps)  # Start at stance height

# Swing phase
for i in range(num_steps):
    phase = foot2_phase[i]
    if phase <= 1.0:  # Swing phase
        swing_fraction = phase
        foot2_x[i] = stride_length * t[i]  # Follow COM forward motion
        # Foot swing: parabolic trajectory from FOOT_Z_STANCE to FOOT_Z_SWING and back
        swing_lift = (FOOT_Z_SWING - FOOT_Z_STANCE) * np.sin(np.pi * swing_fraction)
        foot2_z[i] = FOOT_Z_STANCE + swing_lift
    else:  # Stance phase
        foot2_x[i] = stride_length * t[i]  # Follow COM forward motion
        foot2_z[i] = FOOT_Z_STANCE

foot2_trajectory = np.column_stack([foot2_x, foot2_y, foot2_z])

# --- VERIFY TRAJECTORIES ---
print("=== TRAJECTORY VERIFICATION ===\n")

print("COM Trajectory:")
print(f"  X: {com_trajectory[:, 0].min():.4f} to {com_trajectory[:, 0].max():.4f} (range: {com_trajectory[:, 0].max() - com_trajectory[:, 0].min():.4f})")
print(f"  Y: {com_trajectory[:, 1].min():.4f} to {com_trajectory[:, 1].max():.4f} (range: {com_trajectory[:, 1].max() - com_trajectory[:, 1].min():.4f})")
print(f"  Z: {com_trajectory[:, 2].min():.4f} to {com_trajectory[:, 2].max():.4f} (range: {com_trajectory[:, 2].max() - com_trajectory[:, 2].min():.4f})")

print("\nFoot 1 Trajectory:")
print(f"  X: {foot1_trajectory[:, 0].min():.4f} to {foot1_trajectory[:, 0].max():.4f}")
print(f"  Y: {foot1_trajectory[:, 1].min():.4f} to {foot1_trajectory[:, 1].max():.4f}")
print(f"  Z: {foot1_trajectory[:, 2].min():.4f} to {foot1_trajectory[:, 2].max():.4f} (swing height: {foot1_trajectory[:, 2].max():.4f})")

print("\nFoot 2 Trajectory:")
print(f"  X: {foot2_trajectory[:, 0].min():.4f} to {foot2_trajectory[:, 0].max():.4f}")
print(f"  Y: {foot2_trajectory[:, 1].min():.4f} to {foot2_trajectory[:, 1].max():.4f}")
print(f"  Z: {foot2_trajectory[:, 2].min():.4f} to {foot2_trajectory[:, 2].max():.4f} (swing height: {foot2_trajectory[:, 2].max():.4f})")

print("\n=== SAMPLE FRAMES ===\n")
print("Frame  | COM X   | COM Z   | Foot1 Z | Foot2 Z | Description")
print("-------|---------|---------|---------|---------|------------------")
for i in [0, 50, 100, 150, 200, 250, 300, 350, 399]:
    phase1 = (i / num_steps * 2) % 2
    phase2 = ((i / num_steps * 2 + 1) % 2)
    
    desc1 = "Stance" if phase1 > 1 else "Swing"
    desc2 = "Stance" if phase2 > 1 else "Swing"
    desc = f"F1:{desc1} F2:{desc2}"
    
    print(f"{i:5d}  | {com_trajectory[i, 0]:.4f} | {com_trajectory[i, 2]:.4f} | {foot1_trajectory[i, 2]:.4f}  | {foot2_trajectory[i, 2]:.4f}  | {desc}")

# --- SAVE TRAJECTORIES ---
np.save("com_trajectory.npy", com_trajectory)
np.save("foot1_trajectory.npy", foot1_trajectory)
np.save("foot2_trajectory.npy", foot2_trajectory)

print("\n✅ Trajectories saved:")
print("   - com_trajectory.npy")
print("   - foot1_trajectory.npy")
print("   - foot2_trajectory.npy")
print("\nThese represent realistic walking with:")
print("   - Alternating foot stance/swing phases")
print("   - COM vertical oscillation (bounce)")
print("   - Proper forward progression")
print("   - Foot clearance during swing phase")
