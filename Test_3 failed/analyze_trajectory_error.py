"""
Compare planned (IK) trajectory vs actual foot positions during simulation
"""

import numpy as np
import mujoco
from pathlib import Path
import matplotlib.pyplot as plt

# Load model
model = mujoco.MjModel.from_xml_path('legged_robot_ik.xml')
data = mujoco.MjData(model)

# Load trajectories and IK solutions
traj_file = Path(__file__).parent / "biped_10step_sinusoidal.npy"
traj_data = np.load(str(traj_file), allow_pickle=True).item()
left_trajectory_planned = traj_data['left_trajectory']
right_trajectory_planned = traj_data['right_trajectory']
times = traj_data['times']

ik_file = Path(__file__).parent / "biped_ik_solutions.npy"
ik_data = np.load(str(ik_file), allow_pickle=True).item()
left_angles = ik_data['left_joint_angles']
right_angles = ik_data['right_joint_angles']

# Get foot body IDs
foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'foot_1')
foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'foot_2')

# Track actual positions during simulation
left_trajectory_actual = []
right_trajectory_actual = []

print("Simulating walking and tracking foot positions...")

for i in range(len(times)):
    # Set joint angles
    data.qpos[3] = left_angles[i, 0]
    data.qpos[4] = left_angles[i, 1]
    data.qpos[5] = left_angles[i, 2]
    data.qpos[6] = right_angles[i, 0]
    data.qpos[7] = right_angles[i, 1]
    data.qpos[8] = right_angles[i, 2]
    
    # Forward kinematics
    mujoco.mj_forward(model, data)
    
    # Record actual positions
    foot1_pos = data.xpos[foot1_id]
    foot2_pos = data.xpos[foot2_id]
    
    left_trajectory_actual.append([foot1_pos[0], foot1_pos[1], foot1_pos[2]])
    right_trajectory_actual.append([foot2_pos[0], foot2_pos[1], foot2_pos[2]])
    
    if (i + 1) % 100 == 0:
        print(f"  Frame {i+1}/{len(times)}")

left_trajectory_actual = np.array(left_trajectory_actual)
right_trajectory_actual = np.array(right_trajectory_actual)

# Calculate errors
left_errors = np.linalg.norm(left_trajectory_planned - left_trajectory_actual, axis=1)
right_errors = np.linalg.norm(right_trajectory_planned - right_trajectory_actual, axis=1)

# Print statistics
print("\n" + "="*70)
print("TRAJECTORY ERROR ANALYSIS")
print("="*70)

print(f"\nLEFT LEG ERROR:")
print(f"  Mean error: {left_errors.mean()*1000:.2f}mm")
print(f"  Max error: {left_errors.max()*1000:.2f}mm")
print(f"  Std dev: {left_errors.std()*1000:.2f}mm")

print(f"\nRIGHT LEG ERROR:")
print(f"  Mean error: {right_errors.mean()*1000:.2f}mm")
print(f"  Max error: {right_errors.max()*1000:.2f}mm")
print(f"  Std dev: {right_errors.std()*1000:.2f}mm")

print(f"\nOVERALL:")
print(f"  Mean error: {(left_errors.mean() + right_errors.mean())/2*1000:.2f}mm")
print(f"  Max error: {max(left_errors.max(), right_errors.max())*1000:.2f}mm")

# Check specific phases
swing_indices = [50, 150, 250, 350]  # Mid-swing frames
stance_indices = [0, 100, 200, 300]  # Stance frames

print(f"\nDURING STANCE PHASE:")
for idx in stance_indices:
    left_err = left_errors[idx]*1000
    right_err = right_errors[idx]*1000
    print(f"  Frame {idx}: Left={left_err:.2f}mm, Right={right_err:.2f}mm")

print(f"\nDURING SWING PHASE:")
for idx in swing_indices:
    left_err = left_errors[idx]*1000
    right_err = right_errors[idx]*1000
    print(f"  Frame {idx}: Left={left_err:.2f}mm, Right={right_err:.2f}mm")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Error over time
ax = axes[0, 0]
ax.plot(times, left_errors*1000, 'b-', linewidth=2, label='Left leg', alpha=0.7)
ax.plot(times, right_errors*1000, 'r-', linewidth=2, label='Right leg', alpha=0.7)
ax.axhline(5, color='g', linestyle='--', label='5mm target', alpha=0.7)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Error [mm]')
ax.set_title('Position Error Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Error distribution
ax = axes[0, 1]
ax.hist(left_errors*1000, bins=30, alpha=0.6, label='Left', color='blue', edgecolor='black')
ax.hist(right_errors*1000, bins=30, alpha=0.6, label='Right', color='red', edgecolor='black')
ax.axvline(5, color='g', linestyle='--', label='5mm target')
ax.set_xlabel('Error [mm]')
ax.set_ylabel('Frequency')
ax.set_title('Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Z position comparison (height)
ax = axes[1, 0]
ax.plot(times, left_trajectory_planned[:, 2]*1000, 'b-', linewidth=1.5, label='Planned', alpha=0.7)
ax.plot(times, left_trajectory_actual[:, 2]*1000, 'b--', linewidth=1.5, label='Actual', alpha=0.7)
ax.plot(times, right_trajectory_planned[:, 2]*1000, 'r-', linewidth=1.5, alpha=0.7)
ax.plot(times, right_trajectory_actual[:, 2]*1000, 'r--', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Height Z [mm]')
ax.set_title('Foot Height Comparison (Left=blue, Right=red)')
ax.grid(True, alpha=0.3)
ax.legend()

# X position comparison (forward)
ax = axes[1, 1]
ax.plot(times, left_trajectory_planned[:, 0]*1000, 'b-', linewidth=1.5, label='Planned', alpha=0.7)
ax.plot(times, left_trajectory_actual[:, 0]*1000, 'b--', linewidth=1.5, label='Actual', alpha=0.7)
ax.plot(times, right_trajectory_planned[:, 0]*1000, 'r-', linewidth=1.5, alpha=0.7)
ax.plot(times, right_trajectory_actual[:, 0]*1000, 'r--', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Forward Position X [mm]')
ax.set_title('Foot Forward Position Comparison (Left=blue, Right=red)')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('trajectory_error_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Saved visualization: trajectory_error_analysis.png")

plt.show()
