#!/usr/bin/env python3
"""
Direct Trajectory Playback Simulation

Plays back pre-computed trajectories in MuJoCo without IK.
"""

import numpy as np
import mujoco
from mujoco import viewer
import time

def main():
    """Run simulation with trajectory playback."""
    
    print("\n" + "="*80)
    print("TRAJECTORY PLAYBACK SIMULATION")
    print("="*80)
    
    # Load model
    model_path = "legged_robot_ik.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Load trajectories
    print("\nLoading trajectories...")
    base_traj = np.load("base_trajectory.npy")
    foot1_traj = np.load("foot1_trajectory.npy")
    foot2_traj = np.load("foot2_trajectory.npy")
    
    n_steps = len(base_traj)
    print(f"  Loaded {n_steps} steps")
    
    # Initial configuration - set base position to first trajectory point
    data.qpos[0:3] = base_traj[0]  # x, y position
    data.qpos[2] = 0  # no rotation
    
    # Set initial foot positions through simple forward kinematics
    # Start with reasonable initial guess for joint angles
    data.qpos[3:6] = [0, -np.pi/4, np.pi/4]  # Left leg
    data.qpos[6:9] = [0, -np.pi/4, np.pi/4]  # Right leg
    
    mujoco.mj_forward(model, data)
    
    print("\nStarting simulation...")
    print(f"  Duration: {n_steps} steps")
    
    # Create target trajectory by fitting joint angles
    dt = model.opt.timestep
    frame_count = 0
    
    with viewer.launch_passive(model, data) as v:
        step = 0
        last_q_left = np.array([0, -np.pi/4, np.pi/4])
        last_q_right = np.array([0, -np.pi/4, np.pi/4])
        
        while v.is_running() and step < n_steps:
            # Update base trajectory
            data.qpos[0] = base_traj[step, 0]  # x
            data.qpos[1] = base_traj[step, 1]  # y
            # Don't set z - let gravity handle it or fix it based on foot positions
            
            # Simple inverse kinematics approximation:
            # Move joints toward achieving foot position
            # This is a simplified approach - just interpolate initial poses
            
            # Calculate progress through gait
            progress = step / n_steps
            
            # Left foot height determines left leg pose
            foot1_z = foot1_traj[step, 2]
            if foot1_z < 0.211:  # Stance phase
                # Leg mostly extended
                data.qpos[3:6] = [0, -np.pi/6, np.pi/6]
            else:  # Swing phase
                # Leg flexed for swing
                data.qpos[3:6] = [0, -np.pi/3, np.pi/3]
            
            # Right foot height determines right leg pose  
            foot2_z = foot2_traj[step, 2]
            if foot2_z < 0.211:  # Stance phase
                # Leg mostly extended
                data.qpos[6:9] = [0, -np.pi/6, np.pi/6]
            else:  # Swing phase
                # Leg flexed for swing
                data.qpos[6:9] = [0, -np.pi/3, np.pi/3]
            
            mujoco.mj_forward(model, data)
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            v.sync()
            step += 1
            frame_count += 1
            
            if step % 100 == 0:
                print(f"  Step {step}/{n_steps}")
    
    print(f"\n✓ Simulation complete ({frame_count} frames)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
