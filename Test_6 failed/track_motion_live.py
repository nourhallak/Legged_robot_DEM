#!/usr/bin/env python3
"""
Live motion tracking - real-time visualization of robot walking on sand
"""
import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

# Initialize
mj.mj_resetData(model, data)

# Set initial pose
data.qpos[model.joint("hip_link_2_1").id] = 0.0
data.qpos[model.joint("link_2_1_link_1_1").id] = -0.5
data.qpos[model.joint("link_1_1_foot_1").id] = 0.0
data.qpos[model.joint("hip_link_2_2").id] = 0.0
data.qpos[model.joint("link_2_2_link_1_2").id] = -0.5
data.qpos[model.joint("link_1_2_foot_2").id] = 0.0

mj.mj_forward(model, data)

# Tracking data
max_history = 500
time_hist = deque(maxlen=max_history)
x_hist = deque(maxlen=max_history)
y_hist = deque(maxlen=max_history)
z_hist = deque(maxlen=max_history)

gait_period = 4.0
sim_time = 0.0
max_time = 120.0

def get_push_phase(t, phase_offset=0):
    """Alternating push gait"""
    t_phase = ((t + phase_offset) % gait_period) / gait_period
    
    if t_phase < 0.4:
        progress = t_phase / 0.4
        hip_angle = 0.9 * np.sin(progress * np.pi)
        knee_angle = -0.3 - 0.25 * np.sin(progress * np.pi)
        ankle_angle = 0.25
    else:
        progress = (t_phase - 0.4) / 0.6
        hip_angle = 0.9 - 1.8 * (progress * progress)
        knee_angle = -0.55 + 0.35 * np.sin(progress * np.pi)
        ankle_angle = 0.1
    
    return hip_angle, knee_angle, ankle_angle

# Set up live plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Robot Motion Tracking - Live', fontsize=16, fontweight='bold')

ax1 = axes[0, 0]  # X position
ax2 = axes[0, 1]  # Y position
ax3 = axes[1, 0]  # Z position
ax4 = axes[1, 1]  # XY trajectory

Kp = 1100.0
Kd = 110.0

print("\n" + "=" * 80)
print(" " * 20 + "LIVE MOTION TRACKING - ROBOT WALKING ON SAND")
print("=" * 80)
print("\nPress Ctrl+C to stop\n")

step = 0
update_freq = 50  # Update plot every 50 steps

try:
    while sim_time < max_time:
        # Get gait targets
        left_hip_target, left_knee_target, left_ankle_target = get_push_phase(sim_time, phase_offset=0)
        right_hip_target, right_knee_target, right_ankle_target = get_push_phase(sim_time, phase_offset=gait_period/2)
        
        # Apply PD control
        left_hip_error = left_hip_target - data.qpos[model.joint("hip_link_2_1").id]
        left_hip_vel_error = 0 - data.qvel[model.joint("hip_link_2_1").id]
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = Kp * left_hip_error + Kd * left_hip_vel_error
        
        left_knee_error = left_knee_target - data.qpos[model.joint("link_2_1_link_1_1").id]
        left_knee_vel_error = 0 - data.qvel[model.joint("link_2_1_link_1_1").id]
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = Kp * left_knee_error + Kd * left_knee_vel_error
        
        left_ankle_error = left_ankle_target - data.qpos[model.joint("link_1_1_foot_1").id]
        left_ankle_vel_error = 0 - data.qvel[model.joint("link_1_1_foot_1").id]
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = Kp * left_ankle_error + Kd * left_ankle_vel_error
        
        right_hip_error = right_hip_target - data.qpos[model.joint("hip_link_2_2").id]
        right_hip_vel_error = 0 - data.qvel[model.joint("hip_link_2_2").id]
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = Kp * right_hip_error + Kd * right_hip_vel_error
        
        right_knee_error = right_knee_target - data.qpos[model.joint("link_2_2_link_1_2").id]
        right_knee_vel_error = 0 - data.qvel[model.joint("link_2_2_link_1_2").id]
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = Kp * right_knee_error + Kd * right_knee_vel_error
        
        right_ankle_error = right_ankle_target - data.qpos[model.joint("link_1_2_foot_2").id]
        right_ankle_vel_error = 0 - data.qvel[model.joint("link_1_2_foot_2").id]
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = Kp * right_ankle_error + Kd * right_ankle_vel_error
        
        # Simulate
        mj.mj_step(model, data)
        sim_time += model.opt.timestep
        step += 1
        
        # Record position
        hip_pos = data.body('hip').xpos
        time_hist.append(sim_time)
        x_hist.append(hip_pos[0])
        y_hist.append(hip_pos[1])
        z_hist.append(hip_pos[2])
        
        # Update plot every update_freq steps
        if step % update_freq == 0:
            # Clear axes
            for ax in axes.flat:
                ax.clear()
            
            # Plot X position
            ax1.plot(list(time_hist), list(x_hist), 'b-', linewidth=2)
            ax1.set_ylabel('X Position (m)', fontsize=11, fontweight='bold')
            ax1.set_title('Forward Position', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            if len(x_hist) > 0:
                ax1.text(0.02, 0.95, f'Current: {x_hist[-1]:.4f}m', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Plot Y position
            ax2.plot(list(time_hist), list(y_hist), 'g-', linewidth=2)
            ax2.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='Constraint: ±0.01m')
            ax2.axhline(y=-0.01, color='r', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
            ax2.set_title('Lateral Position (Sand Centering)', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            if len(y_hist) > 0:
                ax2.text(0.02, 0.95, f'Current: {y_hist[-1]:+.6f}m', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            # Plot Z position (height)
            ax3.plot(list(time_hist), list(z_hist), 'r-', linewidth=2)
            ax3.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Z Position (m)', fontsize=11, fontweight='bold')
            ax3.set_title('Height (Stability)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            if len(z_hist) > 0:
                ax3.text(0.02, 0.95, f'Current: {z_hist[-1]:.4f}m', 
                        transform=ax3.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            
            # Plot XY trajectory
            ax4.plot(list(x_hist), list(y_hist), 'purple', linewidth=2, alpha=0.7)
            ax4.scatter([x_hist[-1]], [y_hist[-1]], color='red', s=100, zorder=5, label='Current')
            ax4.scatter([0.150], [0.005], color='green', s=100, zorder=5, marker='s', label='Start')
            ax4.axhline(y=0.01, color='orange', linestyle='--', alpha=0.5)
            ax4.axhline(y=-0.01, color='orange', linestyle='--', alpha=0.5)
            ax4.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
            ax4.set_title('Top-Down Trajectory', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.pause(0.01)
            
            # Console output
            if int(sim_time) > int(sim_time - model.opt.timestep * update_freq):
                displacement = x_hist[-1] - 0.150
                velocity = (x_hist[-1] - x_hist[0]) / sim_time if len(x_hist) > 1 else 0
                print(f"[t={sim_time:7.2f}s] X={x_hist[-1]:.4f}m | Y={y_hist[-1]:+.6f}m | "
                      f"Z={z_hist[-1]:.4f}m | Displ={displacement:.4f}m | Vel={velocity:.4f}m/s")

except KeyboardInterrupt:
    print("\n\n[!] Stopped by user")

print("\n" + "=" * 80)
print("[+] Final Statistics:")
print(f"    Start position: X=0.1500m, Y=0.0050m")
print(f"    Final position: X={x_hist[-1]:.4f}m, Y={y_hist[-1]:+.6f}m")
print(f"    Total displacement: {x_hist[-1] - 0.150:.4f}m")
print(f"    Average velocity: {(x_hist[-1] - 0.150) / sim_time:.4f}m/s")
print(f"    Simulation time: {sim_time:.2f}s")
print("=" * 80 + "\n")

plt.show()
