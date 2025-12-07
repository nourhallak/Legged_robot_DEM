#!/usr/bin/env python3
"""
Optimized gait for sustained walking on sand with hip at Z=0.400m
Using a trotting gait: diagonal legs work together
"""
import mujoco as mj
import numpy as np

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("\n" + "="*80)
print("TROT GAIT TEST (Diagonal Leg Coordination)")
print("="*80)

# Initial pose
data.qpos[model.joint("hip_link_2_1").id] = 0.0
data.qpos[model.joint("link_2_1_link_1_1").id] = -0.6
data.qpos[model.joint("link_1_1_foot_1").id] = 0.0
data.qpos[model.joint("hip_link_2_2").id] = 0.0
data.qpos[model.joint("link_2_2_link_1_2").id] = -0.6
data.qpos[model.joint("link_1_2_foot_2").id] = 0.0

mj.mj_forward(model, data)

sim_time = 0
t_end = 30.0
dt = 0.002

pos_history_x = []
times = []

while sim_time < t_end:
    phase = (sim_time % 1.0) / 1.0  # 0 to 1, cycles every 1 second
    
    if phase < 0.5:  # First half: Legs 1&2 push alternately
        # Push with leg 1 (forward hip drive)
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = 0.4
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = -0.2
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.1
        
        # Lift leg 2 for swing
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = -0.3
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.3
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
        
    else:  # Second half: Legs swap
        # Lift leg 1 for swing
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = -0.3
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.3
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
        
        # Push with leg 2 (forward hip drive)
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = 0.4
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = -0.2
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.1
    
    mj.mj_step(model, data)
    sim_time = data.time
    
    pos_history_x.append(data.body('hip').xpos[0])
    times.append(sim_time)
    
    if int(sim_time * 2) % 10 == 0:  # Every 5 seconds
        dx = pos_history_x[-1] - pos_history_x[0]
        print(f"T={sim_time:6.2f}s | X={data.body('hip').xpos[0]:.4f}m | dX={dx:+.4f}m | VelX={data.body('hip').cvel[3]:.4f}m/s")

print("\n" + "="*80)
print("RESULTS - TROT GAIT")
print("="*80)
dx_total = pos_history_x[-1] - pos_history_x[0]
print(f"Total X displacement: {dx_total:+.4f}m in {sim_time:.1f}s")
print(f"Average velocity: {dx_total/sim_time:+.4f}m/s")
if dx_total > 0.05:
    print("✓ WALKING SUCCESSFUL - Robot achieved sustained forward motion!")
else:
    print("✗ Walking not effective yet")
print("="*80 + "\n")
