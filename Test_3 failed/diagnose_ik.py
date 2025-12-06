"""
Diagnostic: Verify IK Solutions are Correct
=============================================

Check if the IK-solved joint angles actually produce the desired foot positions.
"""

import numpy as np
import mujoco
from pathlib import Path


def forward_kinematics_mujoco(model, data, leg_side='left'):
    """Get foot position using MuJoCo forward kinematics."""
    if leg_side == 'left':
        foot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'foot_1')
    else:
        foot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'foot_2')
    
    if foot_body_id >= 0:
        pos = data.xpos[foot_body_id]
        return np.array(pos)
    return np.array([0, 0, 0])


def set_joint_angles(model, data, left_angles, right_angles):
    """Set joint angles directly (without control)."""
    # Left leg
    data.qpos[3] = left_angles[0]   # hip_link_2_1
    data.qpos[4] = left_angles[1]   # link_2_1_link_1_1
    data.qpos[5] = left_angles[2]   # link_1_1_foot_1
    
    # Right leg
    data.qpos[6] = right_angles[0]  # hip_link_2_2
    data.qpos[7] = right_angles[1]  # link_2_2_link_1_2
    data.qpos[8] = right_angles[2]  # link_1_2_foot_2
    
    # Recompute forward kinematics
    mujoco.mj_forward(model, data)


print("="*70)
print("DIAGNOSTIC: Verify IK Solutions")
print("="*70)

# Load model
model_path = Path(__file__).parent / "legged_robot_ik.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Load trajectories and IK solutions
traj_file = Path(__file__).parent / "biped_10step_sinusoidal.npy"
traj_data = np.load(str(traj_file), allow_pickle=True).item()

ik_file = Path(__file__).parent / "biped_ik_solutions.npy"
ik_data = np.load(str(ik_file), allow_pickle=True).item()

left_foot_traj = traj_data['left_trajectory']
right_foot_traj = traj_data['right_trajectory']

left_angles = ik_data['left_joint_angles']
right_angles = ik_data['right_joint_angles']

print(f"\n📊 Checking IK solutions...")
print(f"   Trajectory points: {len(left_foot_traj)}")
print(f"   IK solutions: {len(left_angles)}")

# Check a few random points
test_indices = [0, 100, 250, 500, 750, 999]

print(f"\n📈 IK Accuracy Check (Target vs Actual foot position):\n")
print(f"{'Idx':<5} {'Target X':<12} {'Actual X':<12} {'Target Z':<12} {'Actual Z':<12} {'Error':<10}")
print("-" * 73)

errors = []

for idx in test_indices:
    # Set joint angles
    set_joint_angles(model, data, left_angles[idx], right_angles[idx])
    
    # Get actual foot position
    left_foot_actual = forward_kinematics_mujoco(model, data, 'left')
    
    # Get target foot position
    left_foot_target = left_foot_traj[idx]
    
    # Calculate error
    error_xz = np.sqrt((left_foot_target[0] - left_foot_actual[0])**2 + 
                       (left_foot_target[2] - left_foot_actual[2])**2)
    errors.append(error_xz)
    
    print(f"{idx:<5} {left_foot_target[0]:<12.4f} {left_foot_actual[0]:<12.4f} {left_foot_target[2]:<12.4f} {left_foot_actual[2]:<12.4f} {error_xz:<10.4f}")

print(f"\n📊 Error Statistics:")
print(f"   Mean error: {np.mean(errors)*1000:.2f} mm")
print(f"   Max error:  {np.max(errors)*1000:.2f} mm")
print(f"   Min error:  {np.min(errors)*1000:.2f} mm")

print(f"\n💡 Analysis:")
if np.mean(errors) < 0.01:
    print(f"   ✅ IK solutions are accurate!")
elif np.mean(errors) < 0.02:
    print(f"   ⚠️  IK solutions have moderate error - acceptable for walking")
else:
    print(f"   ❌ IK solutions have significant error - may cause flying motion")
    print(f"\n   Possible causes:")
    print(f"   1. Trajectory starts from unrealistic position")
    print(f"   2. IK didn't converge properly")
    print(f"   3. Hip height or leg geometry mismatch")
