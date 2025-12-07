#!/usr/bin/env python3
"""
Fix and regenerate trajectories for Test_1 failed folder
- Analyze current broken trajectories
- Generate new smooth walking trajectories
- Compute IK for joint angles
"""

import numpy as np
import mujoco
from scipy.optimize import minimize
import matplotlib.pyplot as plt

print("\n" + "="*80)
print("TRAJECTORY GENERATION AND IK SOLVING FOR TEST_1 FAILED")
print("="*80)

# Load model
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Read XML from Test_1 and modify paths to use parent folder
xml_path = os.path.join(script_dir, "legged_robot_ik.xml")
with open(xml_path, 'r') as f:
    xml_content = f.read()

# Replace relative mesh paths to point to parent
xml_content = xml_content.replace('Legged_robot/meshes/', '../Legged_robot/meshes/')

# Write temporary corrected XML
temp_xml = os.path.join(script_dir, "legged_robot_temp.xml")
with open(temp_xml, 'w') as f:
    f.write(xml_content)

# Load from corrected XML
model = mujoco.MjModel.from_xml_path(temp_xml)
data = mujoco.MjData(model)

# ============================================================================
# STEP 1: Generate proper walking trajectories
# ============================================================================
print("\n" + "="*80)
print("STEP 1: GENERATE WALKING TRAJECTORIES")
print("="*80)

# Gait parameters
STRIDE_LENGTH = 8  # mm forward per cycle
HIP_HEIGHT = 420  # mm (approximate)
STANCE_PHASE = 0.6  # 60% stance, 40% swing
N_STEPS = 300
LEFT_Y = -13.43  # mm (fixed for biped)
RIGHT_Y = -6.4   # mm (fixed for biped)

# Generate gait cycle (one complete step for one leg)
t = np.linspace(0, 1, N_STEPS)

# Hip trajectory (horizontal movement)
# Move forward at constant velocity
base_x = (t * STRIDE_LENGTH)  # Linear forward motion
base_y = np.zeros_like(t) + 0  # No lateral movement
base_z = np.zeros_like(t)  # No vertical rotation for now

base_trajectory = np.column_stack([base_x, base_y, base_z])

# Foot trajectories with swing and stance phases
# Left foot
foot1_x = np.zeros_like(t)
foot1_y = np.full_like(t, LEFT_Y)
foot1_z = np.zeros_like(t)

# Stance phase (first 60% of cycle): sliding backward relative to hip
# Swing phase (last 40%): moving forward in air
for i, ti in enumerate(t):
    phase = ti % 1.0  # Position in current stride cycle
    
    if phase < STANCE_PHASE:
        # Stance: left foot planted, slides back relative to hip
        foot1_x[i] = base_x[i] - 4 * (phase / STANCE_PHASE)
    else:
        # Swing: left foot swings forward
        swing_t = (phase - STANCE_PHASE) / (1 - STANCE_PHASE)
        swing_height = 2 * np.sin(np.pi * swing_t)  # Arc motion
        foot1_x[i] = base_x[i] + 4 * (swing_t - 0.5)
        foot1_z[i] = swing_height

# Right foot (opposite phase)
foot2_x = np.zeros_like(t)
foot2_y = np.full_like(t, RIGHT_Y)
foot2_z = np.zeros_like(t)

for i, ti in enumerate(t):
    phase = (ti + 0.5) % 1.0  # Shifted by 0.5 (opposite phase)
    
    if phase < STANCE_PHASE:
        # Stance: right foot planted
        foot2_x[i] = base_x[i] - 4 * (phase / STANCE_PHASE)
    else:
        # Swing: right foot swings forward
        swing_t = (phase - STANCE_PHASE) / (1 - STANCE_PHASE)
        swing_height = 2 * np.sin(np.pi * swing_t)
        foot2_x[i] = base_x[i] + 4 * (swing_t - 0.5)
        foot2_z[i] = swing_height

# Add Z offset for ground contact
foot1_z += HIP_HEIGHT - 20  # Feet slightly below hip
foot2_z += HIP_HEIGHT - 20

foot1_trajectory = np.column_stack([foot1_x, foot1_y, foot1_z])
foot2_trajectory = np.column_stack([foot2_x, foot2_y, foot2_z])

print(f"Generated {N_STEPS} trajectory steps")
print(f"Hip X range: {base_x.min():.1f} to {base_x.max():.1f} mm")
print(f"Foot1 Z range: {foot1_z.min():.1f} to {foot1_z.max():.1f} mm")
print(f"Foot2 Z range: {foot2_z.min():.1f} to {foot2_z.max():.1f} mm")

# ============================================================================
# STEP 2: Solve IK for joint angles
# ============================================================================
print("\n" + "="*80)
print("STEP 2: INVERSE KINEMATICS SOLVING")
print("="*80)

def forward_kinematics(model, data, q_left, q_right):
    """Get foot positions from joint angles"""
    data.qpos[3:6] = q_left
    data.qpos[6:9] = q_right
    mujoco.mj_forward(model, data)
    foot1_pos = data.site_xpos[0].copy()
    foot2_pos = data.site_xpos[1].copy()
    return foot1_pos, foot2_pos

def ik_objective(q, target_foot1, target_foot2, model, data):
    """Objective function for IK"""
    foot1_pos, foot2_pos = forward_kinematics(model, data, q[:3], q[3:])
    
    err1 = np.linalg.norm(foot1_pos - target_foot1)
    err2 = np.linalg.norm(foot2_pos - target_foot2)
    
    return err1 + err2

# Solve IK for each step
q_left_traj = []
q_right_traj = []
ik_errors = []

# Initial guess from workspace analysis
q_init_left = np.array([-0.5, -1.0, 0.0])
q_init_right = np.array([-0.5, -1.0, 0.0])

for step in range(N_STEPS):
    if step % 50 == 0:
        print(f"  Solving step {step}/{N_STEPS}...", end='')
    
    target_foot1 = foot1_trajectory[step]
    target_foot2 = foot2_trajectory[step]
    
    q_init = np.concatenate([q_init_left, q_init_right])
    
    # Use L-BFGS-B with bounds
    result = minimize(
        ik_objective,
        q_init,
        args=(target_foot1, target_foot2, model, data),
        method='L-BFGS-B',
        bounds=[
            (-np.pi/2, np.pi/4),   # Hip
            (-np.pi/1.5, np.pi/3),  # Knee
            (-np.pi/4, np.pi/4),    # Ankle
            (-np.pi/2, np.pi/4),
            (-np.pi/1.5, np.pi/3),
            (-np.pi/4, np.pi/4)
        ],
        options={'maxiter': 100}
    )
    
    q_sol = result.x
    q_left_traj.append(q_sol[:3])
    q_right_traj.append(q_sol[3:])
    ik_errors.append(result.fun)
    
    # Warm start for next step
    q_init_left = q_sol[:3]
    q_init_right = q_sol[3:]
    
    if step % 50 == 0:
        print(f" Error: {result.fun:.6f}")

q_left_traj = np.array(q_left_traj)
q_right_traj = np.array(q_right_traj)
ik_errors = np.array(ik_errors)

print(f"\nIK Solution Statistics:")
print(f"  Mean error: {ik_errors.mean():.6f} mm")
print(f"  Max error: {ik_errors.max():.6f} mm")
print(f"  Points with <1mm error: {(ik_errors < 1).sum()}/{N_STEPS}")
print(f"  Points with <5mm error: {(ik_errors < 5).sum()}/{N_STEPS}")

# ============================================================================
# STEP 3: Save trajectories
# ============================================================================
print("\n" + "="*80)
print("STEP 3: SAVING TRAJECTORIES")
print("="*80)

np.save(os.path.join(script_dir, 'hip_trajectory.npy'), base_trajectory)
np.save(os.path.join(script_dir, 'foot1_trajectory.npy'), foot1_trajectory)
np.save(os.path.join(script_dir, 'foot2_trajectory.npy'), foot2_trajectory)
np.save(os.path.join(script_dir, 'q_left_feasible.npy'), q_left_traj)
np.save(os.path.join(script_dir, 'q_right_feasible.npy'), q_right_traj)
np.save(os.path.join(script_dir, 'base_feasible.npy'), base_trajectory)

print("Saved trajectories:")
print("  ✓ hip_trajectory.npy")
print("  ✓ foot1_trajectory.npy")
print("  ✓ foot2_trajectory.npy")
print("  ✓ q_left_feasible.npy")
print("  ✓ q_right_feasible.npy")
print("  ✓ base_feasible.npy")

# ============================================================================
# STEP 4: Visualize
# ============================================================================
print("\n" + "="*80)
print("STEP 4: GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Generated Walking Trajectories', fontsize=14)

# Trajectories
axes[0, 0].plot(base_x, label='Hip X')
axes[0, 0].set_title('Hip Position')
axes[0, 0].set_ylabel('X (mm)')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(foot1_z, label='Foot1 Z', color='blue')
axes[0, 1].plot(foot2_z, label='Foot2 Z', color='red')
axes[0, 1].set_title('Foot Heights')
axes[0, 1].set_ylabel('Z (mm)')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[0, 2].plot(foot1_x, label='Foot1 X', color='blue')
axes[0, 2].plot(foot2_x, label='Foot2 X', color='red')
axes[0, 2].set_title('Foot Forward Position')
axes[0, 2].set_ylabel('X (mm)')
axes[0, 2].legend()
axes[0, 2].grid(True)

# Joint angles
axes[1, 0].plot(q_left_traj[:, 0], label='Hip', color='blue')
axes[1, 0].plot(q_left_traj[:, 1], label='Knee', color='green')
axes[1, 0].plot(q_left_traj[:, 2], label='Ankle', color='red')
axes[1, 0].set_title('Left Leg Joint Angles')
axes[1, 0].set_ylabel('Angle (rad)')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(q_right_traj[:, 0], label='Hip', color='blue')
axes[1, 1].plot(q_right_traj[:, 1], label='Knee', color='green')
axes[1, 1].plot(q_right_traj[:, 2], label='Ankle', color='red')
axes[1, 1].set_title('Right Leg Joint Angles')
axes[1, 1].set_ylabel('Angle (rad)')
axes[1, 1].legend()
axes[1, 1].grid(True)

# IK errors
axes[1, 2].plot(ik_errors)
axes[1, 2].axhline(y=5, color='r', linestyle='--', label='5mm threshold')
axes[1, 2].set_title('IK Tracking Error')
axes[1, 2].set_ylabel('Error (mm)')
axes[1, 2].set_xlabel('Step')
axes[1, 2].legend()
axes[1, 2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'trajectories_generated.png'), dpi=100)
print("Saved: trajectories_generated.png")

print("\n" + "="*80)
print("✓ TRAJECTORY GENERATION COMPLETE")
print("="*80 + "\n")
