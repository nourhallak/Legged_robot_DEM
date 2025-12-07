#!/usr/bin/env python3
"""
Top-surface sand walking viewer:
- Robot walks ON TOP of tightly packed sand
- Walks from beginning (X~0.103) to end (X~0.397) of sand bed
- Stops when reaching end of sand
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model with top-surface sand configuration
model = mujoco.MjModel.from_xml_path('legged_robot_sand_top_surface.xml')
data = mujoco.MjData(model)

# Simulation parameters
freq = 0.5  # Hz (5 seconds per cycle)
amplitude = 0.2  # rad (weak control - prevents flying)
duration = 100  # seconds (enough for full walk across sand)

# Sand bed boundaries
sand_x_min = 0.103  # Sand starts here
sand_x_max = 0.397  # Sand ends here
stop_threshold = 0.390  # Stop control when robot reaches this X

# Joint indices for control (4 hip flexion joints)
# Order: front-left, front-right, back-left, back-right
leg_joints = [1, 2, 3, 4]  # Assuming these are the hip flexion joints

# Track gait phase for alternating legs (trotting)
# Front-left and back-right push together, then front-right and back-left
def get_control(t, x_pos):
    """
    Trotting gait control: alternating diagonal legs push backward
    Front-left + back-right pair 1, then front-right + back-left pair 2
    """
    ctrl = np.zeros(model.nu)
    
    # Stop control when reaching end of sand
    if x_pos > stop_threshold:
        return ctrl
    
    # Trotting pattern: alternating diagonal pairs
    phase_global = 2 * np.pi * freq * t
    
    # Pair 1: front-left (leg 0), back-right (leg 3)
    # Pair 2: front-right (leg 1), back-left (leg 2)
    
    phase_pair1 = np.sin(phase_global)
    phase_pair2 = np.sin(phase_global + np.pi)
    
    # Apply control: push backward when phase is positive
    # max(phase, 0) means only push, never pull
    ctrl[leg_joints[0]] = -amplitude * max(phase_pair1, 0)  # front-left
    ctrl[leg_joints[3]] = -amplitude * max(phase_pair1, 0)  # back-right
    
    ctrl[leg_joints[1]] = -amplitude * max(phase_pair2, 0)  # front-right
    ctrl[leg_joints[2]] = -amplitude * max(phase_pair2, 0)  # back-left
    
    return ctrl

# Run simulation with viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    
    start_time = time.time()
    step_count = 0
    last_print_time = 0
    walking_started = False
    
    while viewer.is_running():
        t = time.time() - start_time
        
        if t > duration:
            print("\n✓ Simulation complete")
            break
        
        # Get current robot position (X coordinate)
        # Assuming body 0 is the main body or body with index for "body"
        # Check model for actual body that tracks position
        body_xpos = data.body('torso').xpos if 'torso' in [model.body(i).name for i in range(model.nbody)] else data.xpos[0]
        x_pos = body_xpos[0] if isinstance(body_xpos, np.ndarray) else data.xpos[1][0]
        
        # Check if reached sand start (for timing reference)
        if not walking_started and x_pos > sand_x_min:
            walking_started = True
            print(f"\n✓ Robot reached sand start (X={x_pos:.4f}m)")
        
        # Get control input
        data.ctrl[:] = get_control(t, x_pos)
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Print progress every 5 seconds
        if t - last_print_time >= 5:
            hip_height = data.xpos[1][2] if len(data.xpos) > 1 else 0  # Z coordinate
            vel_x = data.qvel[1] if len(data.qvel) > 1 else 0  # X velocity
            
            # Count sand contacts
            contacts = len([c for c in data.contact if c.dist < 0.001])
            
            status = "✓ WALKING"
            if x_pos > stop_threshold:
                status = "✓ STOPPED (reached sand end)"
            
            print(f"Time {t:6.1f}s | X={x_pos:.4f}m | Hip_Z={hip_height:.4f}m | V={vel_x*1000:.1f}mm/s | Contacts={contacts} | {status}")
            last_print_time = t
            
            # Check if reached end
            if x_pos > sand_x_max:
                print(f"\n✓ Robot reached sand end (X={x_pos:.4f}m) - STOPPING")
                print(f"Total distance walked on sand: {x_pos - sand_x_min:.4f}m")
                break
        
        # Update viewer
        viewer.sync()
        step_count += 1

print(f"\nSimulation ended. Total steps: {step_count}")
print(f"Average frequency: {step_count / duration:.1f} steps/sec")
