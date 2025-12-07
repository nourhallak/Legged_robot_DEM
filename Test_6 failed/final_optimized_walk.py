#!/usr/bin/env python3
"""
Final optimized gait with hip at Z=0.400m
Using repeated side-stepping pushes for consistent forward motion
"""
import mujoco as mj
import numpy as np

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("\n" + "="*80)
print("OPTIMIZED SIDE-PUSH GAIT - WALKING ON SAND AT Z=0.400m")
print("="*80)

# Initial pose - both legs ready
data.qpos[model.joint("hip_link_2_1").id] = 0.0
data.qpos[model.joint("link_2_1_link_1_1").id] = -0.6
data.qpos[model.joint("link_1_1_foot_1").id] = 0.0
data.qpos[model.joint("hip_link_2_2").id] = 0.0
data.qpos[model.joint("link_2_2_link_1_2").id] = -0.6
data.qpos[model.joint("link_1_2_foot_2").id] = 0.0

mj.mj_forward(model, data)

sim_time = 0
t_end = 60.0

pos_history_x = []
times = []
vel_max = 0
vel_samples = []

while sim_time < t_end:
    phase = (sim_time % 2.0) / 2.0  # Cycle every 2 seconds
    
    # Side-by-side pushing - both legs push together forward
    leg1_hip_cmd = 0.5 * np.sin(2 * np.pi * phase)
    leg2_hip_cmd = 0.5 * np.sin(2 * np.pi * phase)
    
    # Knee flex to maintain ground contact
    knee_flex = -0.7 + 0.2 * np.sin(2 * np.pi * phase)
    
    data.ctrl[model.actuator("hip_link_2_1_motor").id] = leg1_hip_cmd
    data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = knee_flex * 0.5
    data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
    
    data.ctrl[model.actuator("hip_link_2_2_motor").id] = leg2_hip_cmd
    data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = knee_flex * 0.5
    data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
    
    mj.mj_step(model, data)
    sim_time = data.time
    
    pos_history_x.append(data.body('hip').xpos[0])
    times.append(sim_time)
    
    vel_x = data.body('hip').cvel[3]
    vel_max = max(vel_max, vel_x)
    vel_samples.append(vel_x)
    
    if int(sim_time * 2) % 10 == 0:  # Every 5 seconds
        dx = pos_history_x[-1] - pos_history_x[0]
        print(f"T={sim_time:6.2f}s | X={data.body('hip').xpos[0]:.4f}m | dX={dx:+.4f}m | VelX={vel_x:.4f}m/s")

print("\n" + "="*80)
print("FINAL RESULTS - SIDE-PUSH GAIT")
print("="*80)
dx_total = pos_history_x[-1] - pos_history_x[0]
avg_vel = np.mean(vel_samples)
print(f"Total X displacement: {dx_total:+.4f}m in {sim_time:.1f}s")
print(f"Average velocity: {dx_total/sim_time:+.4f}m/s")
print(f"Mean velocity (samples): {avg_vel:+.4f}m/s")
print(f"Max velocity: {vel_max:+.4f}m/s")
print("="*80 + "\n")
