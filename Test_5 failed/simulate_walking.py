#!/usr/bin/env python3
"""
Simulate Robot Walking with Solved Joint Angles

Plays back the IK-solved joint trajectories in MuJoCo.
"""

import numpy as np
import mujoco
import time
from mujoco import viewer

def simulate_walking():
    """Simulate robot walking with solved trajectories."""
    
    print("\n" + "="*80)
    print("ROBOT WALKING SIMULATION")
    print("="*80)
    
    # Load model
    model = mujoco.MjModel.from_xml_path("legged_robot_ik.xml")
    # Slow down simulation by reducing timestep (100x slower)
    model.opt.timestep = model.opt.timestep * 0.01
    data = mujoco.MjData(model)
    
    # Load trajectories and joint angles
    print("\nLoading data...")
    base = np.load("base_feasible.npy")
    q_left = np.load("q_left_feasible.npy")
    q_right = np.load("q_right_feasible.npy")
    
    n_steps = len(q_left)
    print(f"  Loaded {n_steps} steps")
    print(f"  Base shape: {base.shape}")
    print(f"  Left leg shape: {q_left.shape}")
    print(f"  Right leg shape: {q_right.shape}")
    
    # Set initial pose
    print("\nInitializing simulation...")
    data.qpos[0:3] = base[0]  # Set base position (x, y, rotation)
    data.qpos[3:6] = q_left[0]
    data.qpos[6:9] = q_right[0]
    
    mujoco.mj_forward(model, data)
    
    print(f"  Initial base position: {data.xpos[1]*1000}")
    print(f"  Initial left foot position: {data.site_xpos[0]*1000}")
    print(f"  Initial right foot position: {data.site_xpos[1]*1000}")
    
    # Run simulation
    print(f"\nRunning simulation (repeating forever)...")
    print("  (Close window to stop)")
    
    dt = model.opt.timestep
    frame = 0
    cycle = 0
    
    with viewer.launch_passive(model, data) as v:
        step = 0
        frame_counter = 0
        
        while v.is_running():
            # Get target joint angles for this step
            q_target_left = q_left[step]
            q_target_right = q_right[step]
            base_target = base[step]
            
            # Set base position directly (kinematic constraint - no actuator)
            data.qpos[0:3] = base_target
            
            # PD control for leg joints
            kp = 100  # Position gain
            kd = 10   # Damping gain
            
            # Left leg control (joints 3,4,5)
            for i in range(3):
                q_error = q_target_left[i] - data.qpos[3 + i]
                qv_error = 0 - data.qvel[3 + i]  # Target velocity is 0
                data.ctrl[i] = kp * q_error + kd * qv_error
            
            # Right leg control (joints 6,7,8)
            for i in range(3):
                q_error = q_target_right[i] - data.qpos[6 + i]
                qv_error = 0 - data.qvel[6 + i]
                data.ctrl[3 + i] = kp * q_error + kd * qv_error
            
            # Physics step
            mujoco.mj_step(model, data)
            v.sync()
            
            # Increment frame counter
            frame_counter += 1
            
            # Move to next trajectory step every N physics steps
            if frame_counter >= 10:  # 10 physics steps per trajectory step
                frame_counter = 0
                step += 1
                if step >= n_steps:
                    step = 0
                    cycle += 1
                    if cycle % 5 == 0:
                        left_pos = data.site_xpos[0]
                        right_pos = data.site_xpos[1]
                        print(f"  Cycle {cycle}: Left Z={left_pos[2]*1000:.1f}mm, Right Z={right_pos[2]*1000:.1f}mm")
            
            # Add delay for slow motion
            time.sleep(0.001)
    
    print(f"\n✓ Simulation stopped ({frame} frames, {cycle} cycles)")
    print("="*80 + "\n")


if __name__ == "__main__":
    simulate_walking()
