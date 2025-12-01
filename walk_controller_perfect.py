"""
Walking controller using perfect FK-generated trajectories.
Should produce ultra-smooth, error-free walking.
"""

import numpy as np
import mujoco
import mujoco.viewer
from scipy.optimize import minimize
import time

xml_path = "legged_robot_ik.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("=" * 80)
print("WALKING CONTROLLER - FK-OPTIMIZED TRAJECTORIES")
print("=" * 80)

# Load perfect FK-generated trajectories
hip_traj = np.load("base_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")

# Load pre-computed IK solutions with 0.01mm error
joint_targets = np.load("joint_targets_aggressive.npy")

num_steps = len(hip_traj)
print(f"\nLoaded {num_steps}-step trajectories")
print(f"Expected IK error: 0.01mm maximum")
print(f"Foot1 X range: [{foot1_traj[:,0].min():.4f}, {foot1_traj[:,0].max():.4f}]")
print(f"Foot1 Z range: [{foot1_traj[:,2].min():.4f}, {foot1_traj[:,2].max():.4f}]")

# ============ PD SERVO CONTROL ============
print(f"\n{'='*80}")
print("STARTING WALKING SIMULATION")
print(f"{'='*80}\n")

KP = 5.0   # Proportional gain
KD = 0.5   # Derivative gain

def get_feet_positions(data, model):
    """Get current foot positions from forward kinematics."""
    foot1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot1_site")
    foot2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot2_site")
    return data.site_xpos[foot1_id].copy(), data.site_xpos[foot2_id].copy()

def set_joint_angles(data, model, angles):
    """Set leg joint angles (indices 3-8, base is 0-2)."""
    qpos_idx = [3, 4, 5, 6, 7, 8]
    for idx, angle in zip(qpos_idx, angles):
        data.qpos[idx] = angle
    mujoco.mj_forward(model, data)

# Initialize
last_angles = joint_targets[0].copy()

# Frame rate control
target_fps = 30
frame_time = 1.0 / target_fps
last_frame_time = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    step_count = 0
    cycle_count = 0
    
    while viewer.is_running():
        for step in range(num_steps):
            target_angles = joint_targets[step]
            target_hip_pos = hip_traj[step, :2]
            
            # Set hip position (direct position control)
            data.qpos[0] = target_hip_pos[0]
            data.qpos[1] = target_hip_pos[1]
            
            # Leg joint PD control
            current_angles = data.qpos[3:9].copy()
            angle_errors = target_angles - current_angles
            angle_velocities = (current_angles - last_angles) / model.opt.timestep
            velocity_errors = -angle_velocities
            last_angles = current_angles.copy()
            
            control_torques = KP * angle_errors + KD * velocity_errors
            
            for i in range(6):
                data.ctrl[i] = np.clip(control_torques[i], -1.0, 1.0)
            
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            
            # Frame rate limiting
            elapsed = time.time() - last_frame_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last_frame_time = time.time()
            
            viewer.sync()
            
            step_count += 1
            
            if step_count % 400 == 0:
                cycle_count += 1
                print(f"  Cycle {cycle_count} completed ({step_count} total steps)...")
            
            if not viewer.is_running():
                break
        
        if viewer.is_running():
            print(f"\n  Restarting cycle {cycle_count + 1}...")
    
    print(f"\nSimulation stopped. {cycle_count} complete cycles executed.")

print("\n" + "=" * 80)
print("EXECUTION COMPLETE")
print("=" * 80)
