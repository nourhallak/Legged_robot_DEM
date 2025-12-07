#!/usr/bin/env python3
"""
Test walking with hip lowered to Z=0.400m (feet should contact sand at Z~0.410m)
Simple alternating leg gait with knee drive
"""
import mujoco as mj
import numpy as np

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("\n" + "="*80)
print("TESTING ROBOT WITH HIP AT Z=0.400m")
print("="*80)

# Initial pose - stand with knees flexed
data.qpos[model.joint("hip_link_2_1").id] = 0.0
data.qpos[model.joint("link_2_1_link_1_1").id] = -0.6
data.qpos[model.joint("link_1_1_foot_1").id] = 0.0
data.qpos[model.joint("hip_link_2_2").id] = 0.0
data.qpos[model.joint("link_2_2_link_1_2").id] = -0.6
data.qpos[model.joint("link_1_2_foot_2").id] = 0.0

mj.mj_forward(model, data)

# Check foot heights
hip_z = data.body('hip').xpos[2]
foot1_z = data.body('foot_1').xpos[2]
foot2_z = data.body('foot_2').xpos[2]

print(f"\nHip Z: {hip_z:.6f}m")
print(f"Foot 1 Z: {foot1_z:.6f}m")
print(f"Foot 2 Z: {foot2_z:.6f}m")
print(f"Sand Z: 0.442m")
print(f"Gap to sand: {0.442 - foot1_z:.6f}m")

if foot1_z < 0.441:
    print("[OK] Feet are BELOW sand height - can contact!")
else:
    print(f"[!] Feet still above sand by {0.442 - foot1_z:.6f}m")
    print(f"    Lower hip further!")

# Simulate walking
print("\n" + "-"*80)
print("SIMULATING WALKING...")
print("-"*80)

sim_time = 0
t_end = 20.0
dt = 0.002

pos_history_x = []
pos_history_z = []
times = []

# Simple alternating gait
phase = 0
while sim_time < t_end:
    phase = (sim_time % 2.0) / 2.0  # 0 to 1, cycles every 2 seconds
    
    if phase < 0.25:  # Leg 1 stance, leg 2 swing
        # Push with leg 1
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = 0.3
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
        
        # Lift leg 2 (swing)
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = -0.3
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.3
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
        
    elif phase < 0.5:  # Transition
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = 0.0
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
        
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = 0.0
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
        
    elif phase < 0.75:  # Leg 2 stance, leg 1 swing
        # Push with leg 2
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = 0.3
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
        
        # Lift leg 1 (swing)
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = -0.3
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.3
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
        
    else:  # Transition
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = 0.0
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
        
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = 0.0
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
    
    mj.mj_step(model, data)
    sim_time = data.time
    
    pos_history_x.append(data.body('hip').xpos[0])
    pos_history_z.append(data.body('hip').xpos[2])
    times.append(sim_time)
    
    if int(sim_time * 2) % 10 == 0:  # Print every 5 seconds
        dx = pos_history_x[-1] - pos_history_x[0]
        print(f"T={sim_time:6.2f}s | X={data.body('hip').xpos[0]:.4f}m | dX={dx:+.4f}m | VelX={data.body('hip').cvel[3]:.4f}m/s")

print("\n" + "="*80)
print("RESULTS")
print("="*80)
dx_total = pos_history_x[-1] - pos_history_x[0]
print(f"Total X displacement: {dx_total:+.4f}m in {sim_time:.1f}s")
print(f"Average velocity: {dx_total/sim_time:+.4f}m/s")
print(f"Final hip height: {pos_history_z[-1]:.6f}m")
print("="*80 + "\n")
