#!/usr/bin/env python3
"""
Aggressive sustained push gait - high force continuous push
Hip at Z=0.400m for good sand contact
"""
import mujoco as mj

model = mj.MjModel.from_xml_path('legged_robot_sand_shifted_low_friction.xml')
data = mj.MjData(model)

print("\n" + "="*80)
print("AGGRESSIVE PUSH GAIT - Z=0.400m (Best contact)")
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

pos_history_x = []

while sim_time < t_end:
    phase = (sim_time % 1.5) / 1.5  # 1.5 second cycle
    
    if phase < 0.4:
        # Leg 1 aggressive push
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = 0.8
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = -0.3
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.2
        
        # Leg 2 lift
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = -0.5
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.5
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
        
    elif phase < 0.7:
        # Both legs hold
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = 0.0
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
        
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = 0.0
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = 0.0
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.0
        
    else:
        # Leg 2 aggressive push
        data.ctrl[model.actuator("hip_link_2_2_motor").id] = 0.8
        data.ctrl[model.actuator("link_2_2_link_1_2_motor").id] = -0.3
        data.ctrl[model.actuator("link_1_2_foot_2_motor").id] = 0.2
        
        # Leg 1 lift
        data.ctrl[model.actuator("hip_link_2_1_motor").id] = -0.5
        data.ctrl[model.actuator("link_2_1_link_1_1_motor").id] = 0.5
        data.ctrl[model.actuator("link_1_1_foot_1_motor").id] = 0.0
    
    mj.mj_step(model, data)
    sim_time = data.time
    pos_history_x.append(data.body('hip').xpos[0])
    
    if int(sim_time * 2) % 10 == 0:
        dx = pos_history_x[-1] - pos_history_x[0]
        print(f"T={sim_time:6.2f}s | X={data.body('hip').xpos[0]:.4f}m | dX={dx:+.4f}m | VelX={data.body('hip').cvel[3]:+.4f}m/s")

print("\n" + "="*80)
dx_total = pos_history_x[-1] - pos_history_x[0]
print(f"Total displacement: {dx_total:+.4f}m in {sim_time:.1f}s")
print(f"Average velocity: {dx_total/sim_time:+.4f}m/s")
print("="*80 + "\n")
