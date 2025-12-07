#!/usr/bin/env python3
"""
Top-surface sand walking simulation (non-interactive):
- Robot walks ON TOP of tightly packed sand
- Walks from beginning (X~0.103) to end (X~0.397) of sand bed
- Stops when reaching end of sand
- Outputs: Motion data, contact info, walking statistics
"""

import mujoco
import numpy as np
import time

# Load model with top-surface sand configuration
model = mujoco.MjModel.from_xml_path('legged_robot_sand_top_surface.xml')
data = mujoco.MjData(model)

# Simulation parameters
freq = 0.5  # Hz (5 seconds per cycle)
amplitude = 0.2  # rad (weak control - prevents flying)
max_duration = 100  # seconds (enough for full walk across sand)

# Sand bed boundaries
sand_x_min = 0.103  # Sand starts here
sand_x_max = 0.397  # Sand ends here
stop_threshold = 0.390  # Stop control when robot reaches this X

# Storage for results
results = {
    'time': [],
    'x_pos': [],
    'z_pos': [],
    'vel_x': [],
    'contacts': []
}

def get_control(t, x_pos):
    """Trotting gait control with auto-stop at sand end"""
    ctrl = np.zeros(model.nu)
    
    if x_pos > stop_threshold:
        return ctrl
    
    # Trotting pattern: alternating diagonal legs
    # Leg 1: indices 0, 1, 2 (hip, thigh, foot)
    # Leg 2: indices 3, 4, 5 (hip, thigh, foot)
    
    phase = np.sin(2.0 * np.pi * freq * t)
    phase_offset = np.sin(2.0 * np.pi * freq * (t + 0.5 / freq))
    
    # Apply weak control to all 6 actuators
    for i in range(model.nu):
        if i < model.nu // 2:
            ctrl[i] = -amplitude * max(phase, 0)
        else:
            ctrl[i] = -amplitude * max(phase_offset, 0)
    
    return ctrl

print("=" * 70)
print("TOP-SURFACE SAND WALKING SIMULATION")
print("=" * 70)
print(f"Gait: Trotting at {freq} Hz, Amplitude: {amplitude} rad")
print(f"Sand region: X=[{sand_x_min:.3f}m, {sand_x_max:.3f}m]")
print(f"Control stops at X={stop_threshold}m (sand end)")
print("=" * 70)

# Run simulation
start_time = time.time()
step_count = 0
last_print_time = 0
walking_started = False
reached_end = False

try:
    while data.time < max_duration:
        # Get current position
        body_idx = 1  # torso/main body
        x_pos = data.xpos[body_idx][0]
        z_pos = data.xpos[body_idx][2]
        vel_x = data.qvel[body_idx] if len(data.qvel) > body_idx else 0
        
        # Track when walking starts
        if not walking_started and x_pos > sand_x_min - 0.01:
            walking_started = True
            print(f"\n✓ WALKING STARTED (X={x_pos:.4f}m)")
        
        # Apply control
        data.ctrl[:] = get_control(data.time, x_pos)
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Count contacts
        contact_count = 0
        for contact in data.contact:
            if contact.dist < 0.001:
                contact_count += 1
        
        # Store results
        results['time'].append(data.time)
        results['x_pos'].append(x_pos)
        results['z_pos'].append(z_pos)
        results['vel_x'].append(vel_x)
        results['contacts'].append(contact_count)
        
        # Print progress every 5 seconds
        if data.time - last_print_time >= 5:
            print(f"T={data.time:6.1f}s | X={x_pos:.4f}m | Z={z_pos:.4f}m | V={vel_x*1000:.1f}mm/s | Contacts={contact_count:2d}")
            last_print_time = data.time
        
        # Check if reached end
        if x_pos > sand_x_max and not reached_end:
            reached_end = True
            print(f"\n✓ REACHED SAND END (X={x_pos:.4f}m)")
            print(f"  Distance walked on sand: {x_pos - sand_x_min:.4f}m")
            print(f"  Time to traverse: {data.time:.1f}s")
            print(f"  Average velocity: {(x_pos - sand_x_min) / data.time * 1000:.1f} mm/s")
            break
        
        step_count += 1

except KeyboardInterrupt:
    print("\nSimulation interrupted by user")

# Final statistics
print("\n" + "=" * 70)
print("WALKING STATISTICS")
print("=" * 70)

if len(results['time']) > 0:
    final_x = results['x_pos'][-1]
    final_z = results['z_pos'][-1]
    final_time = results['time'][-1]
    avg_vel = np.mean(np.abs(results['vel_x'][-20:]))  # Last 20 points
    
    print(f"Total simulation time: {final_time:.1f} seconds")
    print(f"Total steps: {step_count}")
    print(f"Final position: X={final_x:.4f}m, Z={final_z:.4f}m")
    print(f"Robot walked: {final_x - sand_x_min:.4f}m across sand bed")
    print(f"Average velocity (final): {avg_vel*1000:.1f} mm/s")
    
    avg_contacts = np.mean(results['contacts'][-20:])  # Last 20 steps
    print(f"Average sand contacts per frame: {avg_contacts:.1f}")
    
    if reached_end:
        print("\n✓ MISSION ACCOMPLISHED: Robot walked from beginning to end of sand bed")
    else:
        print(f"\n⚠ Robot did not reach sand end (stopped at X={final_x:.4f}m)")

# Check if walking is on top surface
if len(results['z_pos']) > 10:
    avg_z = np.mean(results['z_pos'][-10:])
    print(f"\nHip height (average, last 10 steps): {avg_z:.4f}m")
    print(f"Sand top surface: ~0.435m")
    if avg_z > 0.480:
        print("⚠ WARNING: Robot may be flying/bouncing off sand (Z too high)")
    elif avg_z < 0.420:
        print("⚠ WARNING: Robot may be sinking into sand (Z too low)")
    else:
        print("✓ Robot is walking at appropriate height")

print("=" * 70)
