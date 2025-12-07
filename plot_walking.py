#!/usr/bin/env python3
"""
Analyze and visualize robot walking motion - save plots only
"""
import mujoco as mj
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

# Initial pose
data.qpos[model.joint("hip_link_2_1").id] = 0.0
data.qpos[model.joint("link_2_1_link_1_1").id] = -0.6
data.qpos[model.joint("link_1_1_foot_1").id] = 0.0
data.qpos[model.joint("hip_link_2_2").id] = 0.0
data.qpos[model.joint("link_2_2_link_1_2").id] = -0.6
data.qpos[model.joint("link_1_2_foot_2").id] = 0.0

mj.mj_forward(model, data)

print("Recording walking motion...")

sim_time = 0
t_end = 20.0

# Store trajectory data
times = []
hip_x_pos = []
hip_z_pos = []
foot1_z_pos = []
foot2_z_pos = []
velocities = []
knee1_angles = []
knee2_angles = []

while sim_time < t_end:
    phase = (sim_time % 2.0) / 2.0
    
    # Leg 1 control
    if phase < 0.25:
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = 0.3
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
    elif phase < 0.5:
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = 0.0
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
    else:
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = -0.3
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.3
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
    
    # Leg 2 control (opposite phase)
    if phase < 0.25:
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = -0.3
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.3
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
    elif phase < 0.5:
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = 0.0
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
    else:
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = 0.3
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
    
    mj.mj_step(model, data)
    sim_time = data.time
    
    if int(sim_time * 1000) % 50 == 0:  # Sample every 50ms
        hip = data.body('hip').xpos
        foot1 = data.body('foot_1').xpos
        foot2 = data.body('foot_2').xpos
        
        times.append(sim_time)
        hip_x_pos.append(hip[0])
        hip_z_pos.append(hip[2])
        foot1_z_pos.append(foot1[2])
        foot2_z_pos.append(foot2[2])
        velocities.append(data.body('hip').cvel[3])
        knee1_angles.append(data.qpos[model.joint("link_2_1_link_1_1").id])
        knee2_angles.append(data.qpos[model.joint("link_2_2_link_1_2").id])

print(f"Recorded {len(times)} samples")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Legged Robot Walking on Sand - Motion Analysis', fontsize=14, fontweight='bold')

# Plot 1: Hip X displacement
ax = axes[0, 0]
ax.plot(times, hip_x_pos, 'b-', linewidth=2.5)
ax.fill_between(times, hip_x_pos[0], hip_x_pos, alpha=0.3)
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('X Position (m)', fontsize=11)
ax.set_title('Hip Forward Motion', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
total_displacement = hip_x_pos[-1] - hip_x_pos[0]
ax.text(0.5, 0.95, f'Total: {total_displacement*100:.2f} cm', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 2: Vertical positions
ax = axes[0, 1]
ax.plot(times, hip_z_pos, 'b-', linewidth=2.5, label='Hip Z')
ax.plot(times, foot1_z_pos, 'r--', linewidth=2, label='Foot 1 Z')
ax.plot(times, foot2_z_pos, 'g--', linewidth=2, label='Foot 2 Z')
ax.axhline(y=0.442, color='brown', linestyle=':', linewidth=2.5, label='Sand surface')
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Z Position (m)', fontsize=11)
ax.set_title('Vertical Positions - Feet vs Sand', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_ylim([0.38, 0.48])

# Plot 3: Forward velocity
ax = axes[1, 0]
ax.plot(times, velocities, 'g-', linewidth=2.5)
ax.fill_between(times, 0, velocities, alpha=0.3, color='green')
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Velocity (m/s)', fontsize=11)
ax.set_title('Forward Velocity Profile', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
peak_vel = max(velocities)
ax.text(0.5, 0.95, f'Peak: {peak_vel:.4f} m/s', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Plot 4: Knee joint angles
ax = axes[1, 1]
ax.plot(times, knee1_angles, 'b-', linewidth=2.5, label='Leg 1 Knee')
ax.plot(times, knee2_angles, 'r-', linewidth=2.5, label='Leg 2 Knee')
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Joint Angle (rad)', fontsize=11)
ax.set_title('Knee Joint Angles', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('walking_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: walking_analysis.png")

# Print summary
print("\n" + "="*80)
print("WALKING MOTION SUMMARY")
print("="*80)
print(f"Total displacement: {total_displacement:.4f} m ({total_displacement*100:.2f} cm)")
print(f"Simulation time: {times[-1]:.2f} s")
print(f"Average velocity: {total_displacement/times[-1]:.6f} m/s")
print(f"Peak velocity: {peak_vel:.4f} m/s")
min_foot_z = min(foot1_z_pos + foot2_z_pos)
print(f"Minimum foot height: {min_foot_z:.6f} m")
print(f"Sand surface height: 0.442000 m")
print(f"Contact achieved: {'YES' if min_foot_z < 0.442 else 'NO'}")
print("="*80 + "\n")
