"""
Debug viewer - Check robot motion without viewer
"""

import numpy as np
import mujoco
from pathlib import Path

# Load model
model_path = Path(__file__).parent / "legged_robot_ik.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

print("="*70)
print("BIPED ROBOT DEBUG - Testing Joint Control")
print("="*70)

print(f"\n📐 Model Info:")
print(f"   Bodies: {model.nbody}")
print(f"   Joints: {model.nq}")
print(f"   Actuators: {model.nu}")
print(f"   Timestep: {model.opt.timestep}")

print(f"\n🔍 Joint Details:")
for i in range(min(9, model.nq)):
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"   Joint {i}: {jnt_name} -> qpos index {i}")

print(f"\n🔌 Actuator Details:")
for i in range(min(6, model.nu)):
    act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    jnt_id = model.actuator_trnid[i, 0]
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
    print(f"   Actuator {i}: {act_name} -> Joint {jnt_id} ({jnt_name})")

# Load trajectories
print(f"\n📂 Loading data...")
traj_file = Path(__file__).parent / "biped_10step_sinusoidal.npy"
traj_data = np.load(str(traj_file), allow_pickle=True).item()

ik_file = Path(__file__).parent / "biped_ik_solutions.npy"
ik_data = np.load(str(ik_file), allow_pickle=True).item()

left_angles = ik_data['left_joint_angles']
right_angles = ik_data['right_joint_angles']

print(f"✅ Loaded {len(left_angles)} trajectory points")

# Test simple motion
print(f"\n🧪 Testing Control - Setting joints to target angles...")

# Set initial target angles (first frame)
target_left = left_angles[0]
target_right = right_angles[0]

print(f"\n   Target angles (first frame):")
print(f"   Left:  hip={np.degrees(target_left[0]):.1f}° knee={np.degrees(target_left[1]):.1f}° ankle={np.degrees(target_left[2]):.1f}°")
print(f"   Right: hip={np.degrees(target_right[0]):.1f}° knee={np.degrees(target_right[1]):.1f}° ankle={np.degrees(target_right[2]):.1f}°")

# Apply PD control
kp = 100.0
kd = 10.0

motor_names = [
    'hip_link_2_1_motor',
    'link_2_1_link_1_1_motor',
    'link_1_1_foot_1_motor',
    'hip_link_2_2_motor',
    'link_2_2_link_1_2_motor',
    'link_1_2_foot_2_motor',
]

targets = [target_left[0], target_left[1], target_left[2],
          target_right[0], target_right[1], target_right[2]]

print(f"\n   Applying PD control for 100 steps...")

for step in range(100):
    for motor_idx, motor_name in enumerate(motor_names):
        motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_name)
        if motor_id >= 0:
            joint_id = model.actuator_trnid[motor_id, 0]
            if joint_id >= 0 and joint_id < len(data.qpos):
                current_angle = data.qpos[joint_id]
                current_vel = data.qvel[joint_id] if joint_id < len(data.qvel) else 0.0
                target_angle = targets[motor_idx]
                
                error = target_angle - current_angle
                control = kp * error - kd * current_vel
                control = np.clip(control, -1.0, 1.0)
                
                data.ctrl[motor_id] = control
    
    mujoco.mj_step(model, data)
    
    if step % 20 == 0:
        print(f"   Step {step:3d}: ", end="")
        for i in range(3):
            print(f"J{i}={np.degrees(data.qpos[i+3]):.1f}° ", end="")
        print()

print(f"\n✅ Final joint angles (after 100 steps):")
for i in range(6):
    print(f"   qpos[{i}] = {data.qpos[i]:.4f} rad = {np.degrees(data.qpos[i]):.2f}°")

print(f"\n💡 If angles changed from initial values, joint control is working!")
print(f"   If all zeros, there may be an issue with the model or joints.")
