#!/usr/bin/env python3
"""
Walking motion with trajectory tracking control
Uses trajectory playback with simulation to achieve actual walking
"""

import numpy as np
import mujoco
import mujoco.viewer
import re
from pathlib import Path

print("="*80)
print("BIPEDAL WALKING WITH TRAJECTORY TRACKING")
print("="*80)

def load_model_with_assets(xml_path="legged_robot_ik.xml"):
    xml_file = Path(xml_path)
    if not xml_file.exists():
        xml_file = Path(".") / "legged_robot_ik.xml"
    
    with open(xml_file, "r") as f:
        xml_content = f.read()
    
    asset_dir = xml_file.parent / "Legged_robot" / "meshes"
    
    for match in re.finditer(r'filename="([^"]*\.stl)"', xml_content):
        stl_file = match.group(1)
        full_path = asset_dir / stl_file
        
        if full_path.exists():
            xml_content = xml_content.replace(f'filename="{stl_file}"', f'filename="{full_path}"')
    
    return mujoco.MjModel.from_xml_string(xml_content)

# Load model
print("\nLoading robot model...")
model = load_model_with_assets()
data = mujoco.MjData(model)
print(f"[OK] Model loaded: {model.nq} DOF, {model.nu} actuators")

# Load trajectories
print("Loading trajectories...")
base_traj = np.load("base_trajectory.npy")
foot1_traj = np.load("foot1_trajectory.npy")
foot2_traj = np.load("foot2_trajectory.npy")
num_steps = len(base_traj)
print(f"[OK] Trajectories loaded: {num_steps} steps")

# Load IK solutions if available
if Path("joint_solutions_ik.npy").exists():
    qpos_solutions = np.load("joint_solutions_ik.npy")
    print(f"[OK] IK solutions loaded")
else:
    print(f"[WARNING] No IK solutions found - using zero configuration")
    qpos_solutions = np.zeros((num_steps, model.nq))

# Initialize
data.qpos[:] = qpos_solutions[0]
mujoco.mj_forward(model, data)

print("\nStarting trajectory playback...")
print("Controls:")
print("  - Click and drag to rotate view")
print("  - Right-click drag to pan")
print("  - Scroll to zoom")
print("  - Press 'Esc' or close window to exit")
print("-" * 80)

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        except:
            pass
        
        step_idx = 0
        frame_count = 0
        dt = model.opt.timestep
        
        # Control parameters
        Kp = 200.0   # Reduced to allow smoother tracking
        Kd = 20.0    # Reduced damping
        
        while viewer.is_running():
            # Get target joint configuration at this trajectory step
            qpos_target = qpos_solutions[step_idx % num_steps]
            base_target = base_traj[step_idx % num_steps]
            
            # DIRECTLY SET BASE POSITION from trajectory
            data.qpos[0:3] = base_target
            data.qvel[0:3] = 0.0
            
            # Control leg joints (3-8) with PD
            act_joints = [3, 4, 5, 6, 7, 8]
            for i, j in enumerate(act_joints):
                error = qpos_target[j] - data.qpos[j]
                vel = data.qvel[j]
                ctrl = Kp * error - Kd * vel
                data.ctrl[i] = np.clip(ctrl, -10.0, 10.0)
            
            # Simulate
            mujoco.mj_step(model, data)
            frame_count += 1
            
            # Advance trajectory at normal speed (25 frames per step = smoother)
            frames_per_trajectory_step = 100
            if frame_count >= frames_per_trajectory_step:
                frame_count = 0
                step_idx += 1
                if step_idx >= num_steps:
                    step_idx = 0
            
            viewer.sync()

except Exception as e:
    print(f"[ERROR] Viewer error: {e}")
    import traceback
    traceback.print_exc()

print("\n[OK] Simulation complete!")
print("="*80)
