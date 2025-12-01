"""
Physics-based walking controller.
Hip moves forward naturally from leg forces, not direct position control.
"""

import numpy as np
import mujoco
import mujoco.viewer
import time

xml_path = "legged_robot_ik.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("=" * 80)
print("PHYSICS-BASED WALKING (HIP MOVES FROM LEG FORCES)")
print("=" * 80)

# Load trajectories and solutions
hip_traj = np.load("base_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")
joint_targets = np.load("joint_targets_warmstart.npy")

num_steps = len(hip_traj)
print(f"\nLoaded {num_steps}-step trajectory with IK error < 2.6mm")
print(f"Stride: {(foot1_traj[-1, 0] - foot1_traj[0, 0])*1000:.2f}mm")

def get_feet_positions(data, model):
    foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot1_site")
    foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot2_site")
    return data.site_xpos[foot1_id].copy(), data.site_xpos[foot2_id].copy()

def get_hip_position(data):
    hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip")
    return data.xpos[hip_id].copy()

print(f"\n{'='*80}")
print("STARTING PHYSICS-BASED WALKING WITH VIEWER")
print("(Hip movement emerges from leg forces pushing ground)")
print(f"{'='*80}\n")

# PD control gains for joint tracking
KP = 20.0  # Strong proportional gain to push against ground
KD = 2.0   # Derivative gain

# Initialize
step_idx = 0
repeat = 0
max_repeat = 5

# Statistics
errors_foot1 = []
errors_foot2 = []
hip_positions = []
last_angles = joint_targets[0].copy()

target_fps = 30
frame_time = 1.0 / target_fps
last_frame_time = time.time()

# Set initial hip position to avoid damping
data.qpos[0] = 0.0  # Hip X
data.qpos[1] = -0.005  # Hip Y (centered between feet)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    sim_step = 0
    
    while viewer.is_running() and repeat < max_repeat:
        
        # Get current trajectory target
        traj_idx = sim_step % num_steps
        target_angles = joint_targets[traj_idx]
        target_foot1 = foot1_traj[traj_idx]
        target_foot2 = foot2_traj[traj_idx]
        
        # ============ LEG JOINT CONTROL ============
        # Track IK joint targets with PD control
        # Hip position emerges naturally from this tracking
        current_angles = data.qpos[3:9].copy()
        angle_errors = target_angles - current_angles
        angle_velocities = (current_angles - last_angles) / model.opt.timestep
        velocity_errors = -angle_velocities
        
        # PD control on leg joints (this creates forces that push the ground)
        control = KP * angle_errors + KD * velocity_errors
        data.ctrl[:] = np.clip(control, -1.0, 1.0)
        
        last_angles = current_angles.copy()
        
        # ============ SIMULATION ============
        # Physics engine handles ground contact and forces
        mujoco.mj_step(model, data)
        sim_step += 1
        
        # Record data
        actual_foot1, actual_foot2 = get_feet_positions(data, model)
        actual_hip = get_hip_position(data)
        
        error_f1 = np.linalg.norm(actual_foot1 - target_foot1)
        error_f2 = np.linalg.norm(actual_foot2 - target_foot2)
        
        errors_foot1.append(error_f1)
        errors_foot2.append(error_f2)
        hip_positions.append(actual_hip.copy())
        
        # Update repeat counter
        if sim_step % num_steps == 0 and sim_step > 0:
            repeat += 1
            if repeat < max_repeat:
                print(f"\n--- Completed repeat {repeat}/{max_repeat} ---")
                avg_f1 = np.mean(errors_foot1[-num_steps:])
                print(f"    Foot1 tracking error: {avg_f1*1000:.2f}mm")
                print(f"    Hip X position: {actual_hip[0]:.6f}m")
        
        # Frame rate control
        time_now = time.time()
        time_until_next_frame = frame_time - (time_now - last_frame_time)
        if time_until_next_frame > 0:
            time.sleep(time_until_next_frame)
        last_frame_time = time.time()
        
        viewer.sync()
        
        # Periodic status
        if (sim_step % 100) == 0:
            avg_error_1 = np.mean(errors_foot1[-100:]) if len(errors_foot1) >= 100 else np.mean(errors_foot1)
            print(f"Step {sim_step}: Foot1 avg error: {avg_error_1*1000:.2f}mm, Hip X: {actual_hip[0]:.6f}m")

print(f"\n{'='*80}")
print("WALKING SIMULATION COMPLETE")
print(f"{'='*80}\n")

errors_foot1 = np.array(errors_foot1)
errors_foot2 = np.array(errors_foot2)
hip_positions = np.array(hip_positions)

print(f"Tracking Performance:")
print(f"  Foot1 - Mean: {np.mean(errors_foot1)*1000:.2f}mm, Max: {np.max(errors_foot1)*1000:.2f}mm")
print(f"  Foot2 - Mean: {np.mean(errors_foot2)*1000:.2f}mm, Max: {np.max(errors_foot2)*1000:.2f}mm")

print(f"\nHip Forward Progression:")
print(f"  Start X: {hip_positions[0, 0]:.6f}m")
print(f"  End X:   {hip_positions[-1, 0]:.6f}m")
print(f"  Total forward motion: {(hip_positions[-1, 0] - hip_positions[0, 0])*1000:.2f}mm")
print(f"  Average per cycle: {(hip_positions[-1, 0] - hip_positions[0, 0])*1000/max_repeat:.2f}mm")

print(f"\nZ Height variation:")
print(f"  Min: {hip_positions[:, 2].min():.6f}m")
print(f"  Max: {hip_positions[:, 2].max():.6f}m")
print(f"  Range: {(hip_positions[:, 2].max() - hip_positions[:, 2].min())*1000:.2f}mm")

if (hip_positions[-1, 0] - hip_positions[0, 0]) > 0.001:
    print(f"\n✓ Robot WALKING - Hip moved forward!")
else:
    print(f"\n✗ Robot NOT walking - Hip barely moved")
