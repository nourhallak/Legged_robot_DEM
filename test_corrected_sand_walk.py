#!/usr/bin/env python3
"""Test walking with corrected sand layer (feet ON TOP of sand)."""
import mujoco
import numpy as np
import time

# Load the corrected model
model = mujoco.MjModel.from_xml_path('legged_robot_sand_fixed_surface.xml')
data = mujoco.MjData(model)

# Get joint IDs for all joints
joint_names = [model.joint(i).name for i in range(model.nq)]
print(f"\nModel loaded with {model.nq} joint DOFs")
print(f"Number of actuators: {len(data.ctrl)}")
print(f"Joint names: {joint_names[:10]}")

# Initialize data
mujoco.mj_step(model, data)

# Parameters for walking gait
sim_time = 25.0  # seconds
dt = model.opt.timestep  # 0.002
steps = int(sim_time / dt)

# Simple walking: apply torques to leg joints
frequency = 1.0  # Hz
amplitude = 0.3  # radians

# Storage
times = []
positions = []
hip_x = []
hip_z = []
foot1_heights = []
foot2_heights = []
contacts = []

print("\nStarting walking simulation (25 seconds)...")
print("Hip Z position: 0.438m")
print("Expected feet contact: Z ≈ 0.460m (on sand surface)")
print()

# Only use available actuators
num_actuators = len(data.ctrl)
actuator_joints = list(range(min(3, len(joint_names)), min(3 + num_actuators, len(joint_names))))

for step in range(steps):
    t = step * dt
    
    # Simple walking gait: sinusoidal actuation
    phase = np.sin(2 * np.pi * frequency * t)
    
    # Apply torques to available actuators
    for i in range(num_actuators):
        data.ctrl[i] = amplitude * phase
    
    # Step simulation
    mujoco.mj_step(model, data)
    
    # Record data
    if step % 10 == 0:  # Every 0.02 seconds
        times.append(t)
        hip_x.append(data.xpos[model.body('hip').id][0])
        hip_z.append(data.xpos[model.body('hip').id][2])
        
        # Get foot heights (foot bodies)
        try:
            foot1_z = data.xpos[model.body('foot_1').id][2]
            foot1_heights.append(foot1_z)
        except:
            foot1_heights.append(0)
        
        try:
            foot2_z = data.xpos[model.body('foot_2').id][2]
            foot2_heights.append(foot2_z)
        except:
            foot2_heights.append(0)
        
        # Count contacts
        contacts.append(data.ncon)

# Analysis
hip_x = np.array(hip_x)
hip_z = np.array(hip_z)
foot1_heights = np.array(foot1_heights)
foot2_heights = np.array(foot2_heights)
times = np.array(times)

total_displacement = hip_x[-1] - hip_x[0]
avg_velocity = total_displacement / sim_time if sim_time > 0 else 0
max_velocity = np.max(np.diff(hip_x)) / 0.02 if len(hip_x) > 1 else 0

foot1_min = np.min(foot1_heights)
foot1_max = np.max(foot1_heights)
foot2_min = np.min(foot2_heights)
foot2_max = np.max(foot2_heights)

print("="*60)
print("WALKING TEST RESULTS - CORRECTED SAND LAYER")
print("="*60)
print(f"Simulation time: {sim_time} seconds")
print(f"Total displacement: {total_displacement:.4f} m ({total_displacement*100:.2f} cm)")
print(f"Average velocity: {avg_velocity:.6f} m/s")
print(f"Max velocity: {max_velocity:.6f} m/s")
print()
print("FOOT HEIGHTS (should be above Z=0.453m sand surface):")
print(f"  Foot 1: min={foot1_min:.6f}m, max={foot1_max:.6f}m")
print(f"  Foot 2: min={foot2_min:.6f}m, max={foot2_max:.6f}m")
print()
print("SAND SURFACE CHECK:")
sand_surface = 0.453  # Top of sand layer
foot1_above = foot1_min > sand_surface
foot2_above = foot2_min > sand_surface
print(f"  Foot 1 above sand? {foot1_above} (min={foot1_min:.6f} > {sand_surface})")
print(f"  Foot 2 above sand? {foot2_above} (min={foot2_min:.6f} > {sand_surface})")
if foot1_above and foot2_above:
    print("  ✓ CORRECT: Robot walking ON TOP of sand surface!")
else:
    print("  ✗ PROBLEM: Robot penetrating sand layer")
print()
print(f"Average contacts per frame: {np.mean(contacts):.1f}")
print("="*60)

# Save results
np.save('walk_corrected_times.npy', times)
np.save('walk_corrected_hip_x.npy', hip_x)
np.save('walk_corrected_hip_z.npy', hip_z)
np.save('walk_corrected_foot1_heights.npy', foot1_heights)
np.save('walk_corrected_foot2_heights.npy', foot2_heights)

print("\nData saved. Ready to visualize with plot script.")

